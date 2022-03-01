/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cxxabi.h>
#include <execinfo.h>
#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "aten/core/caching_allocator.h"
#include "aten/device/queue.h"
#include "aten/device/mlu_guard.h"
#include "aten/util/python_interface.h"

#if USE_PROFILE
#include "aten/profile/profile_mlu.h"
#endif

#define MASK_WORDS 1

static bool native_memory_strategy = false;

namespace torch_mlu {

// the constant parameters for chunk
constexpr size_t minimum_round_size =
    512;  // all chunks are rounded at least 512 bytes
constexpr size_t small_allocation_size =
    1048576;  // maximum for "small" allocation is 1 Mib
constexpr size_t small_buffer_size =
    2097152;  // "small" allocations are in 2 Mibs chunks
constexpr size_t large_allocation_size =
    10485760;  // allocation sizes from 1 Mib to 10 Mibs use larger chunks
constexpr size_t large_buffer_size =
    20971520;  // "large" allocations may be in 20 Mibs chunks
constexpr size_t maximum_round_size =
    2097152;  // all chunks are rounded at most 2 Mibs

// DEBUG_MODE: the mask size
constexpr size_t mask_bytes = MASK_WORDS * sizeof(int64_t);

// DEBUG_MODE: Debugging Flag
static bool debug_mode = false;

// DEBUG_MODE: backtrace layers
constexpr int layer_num = 10;

bool get_memory_strategy() { return native_memory_strategy; }

void set_memory_strategy(bool ms) {
  native_memory_strategy = ms;
}

std::shared_ptr<int64_t> newMask(int64_t magic_word) {
  std::shared_ptr<int64_t> m (new int64_t[MASK_WORDS], std::default_delete<int64_t[]>());
  for (int i = 0; i < MASK_WORDS; ++i) {
    m.get()[i] = magic_word;
  }
  return m;
}

std::shared_ptr<int64_t> header_mask = newMask(0x4c4955595558494e);
std::shared_ptr<int64_t> footer_mask = newMask(0x48574a4341544348);

struct MemoryStats {
  uint64_t   allocated_size;      // total size allocated in bytes
  uint64_t   max_allocated_size;  // max total size allocated in bytes
  uint64_t   cached_size;         // total size in cache in bytes
  uint64_t   max_cached_size;     // max total size in cache in bytes

  MemoryStats() :
      allocated_size(0), max_allocated_size(0),
      cached_size(0), max_cached_size(0) { }

  virtual void allocated(size_t num_allocated) {
    allocated_size += num_allocated;
    max_allocated_size = std::max(max_allocated_size, allocated_size);
  }

  virtual void deallocated(size_t num_allocated) {
    allocated_size -= num_allocated;
  }

  virtual void cached(size_t num_cached) {
    cached_size += num_cached;
    max_cached_size = std::max(max_cached_size, cached_size);
  }

  virtual void decached(size_t num_cached) {
    cached_size -= num_cached;
  }
};

struct DebugStats : public MemoryStats {
  void allocated(size_t num_allocated) override {
    allocated_size += num_allocated;
    allocated_size -= 2 * mask_bytes;
    max_allocated_size = std::max(max_allocated_size, allocated_size);
  }

  void deallocated(size_t num_allocated) override {
    allocated_size -= num_allocated;
    allocated_size += 2 * mask_bytes;
  }

  void cached(size_t num_cached) override {
    cached_size += num_cached;
    cached_size -= 2 * mask_bytes;
    max_cached_size = std::max(max_cached_size, cached_size);
  }

  void decached(size_t num_cached) override {
    cached_size -= num_cached;
    cached_size += 2 * mask_bytes;
  }
};



// ChunkPool is a sorted list of Chunk, using pointer for comparing
struct Chunk;
typedef bool (*Comparison)(const Chunk*, const Chunk*);
typedef std::set<Chunk*, Comparison> ChunkPool;
using queue_set = std::unordered_set<torch_mlu::Queue>;

struct Chunk {
  int device_id;      // mlu device id
  cnrtQueue_t queue;  // allocation queue
  queue_set queue_in_use;  // queues on which the chunk was used
  size_t size;         // chunk size in bytes
  ChunkPool* pool;     // owning memory pool
  void* ptr;           // memory address
  bool allocated;      // is_allocated flag
  Chunk* prev;         // prev chunk if split from a larger allocation
  Chunk* next;         // next chunk if split from a larger allocation
  int notifier_count;  // number of outstanding MLU notifiers.
  Chunk(int device_id, cnrtQueue_t queue, size_t size, ChunkPool* pool,
        void* ptr)
      : device_id(device_id),
        queue(queue),
        size(size),
        pool(pool),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        queue_in_use(),
        notifier_count(0) {}

  // constructor for search key
  Chunk(int device_id, cnrtQueue_t queue, size_t size)
      : device_id(device_id),
        queue(queue),
        size(size),
        pool(nullptr),
        ptr(nullptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        queue_in_use(),
        notifier_count(0) {}
};

static bool ChunkComparator(const Chunk* a, const Chunk* b) {
  if (a->device_id != b->device_id) {
    return a->device_id < b->device_id;
  }
  if (a->queue != b->queue) {
    return (uintptr_t)a->queue < (uintptr_t)b->queue;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

// format size(byte) in string
static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

struct CachingAllocator {
  // Memory statistics
  std::vector<MemoryStats> memory_stats;

  // lock around all operations
  std::recursive_mutex base_mutex;

  // lock around calls to cnrtFree (to prevent deadlocks with CNCL)
  std::mutex mlu_mutex;

  // cached chunks are larger than 1 MB
  ChunkPool large_chunks;

  // cached chunks are 1 MB or smaller
  ChunkPool small_chunks;

  // allocated chunks by device pointer
  std::unordered_map<void*, Chunk*> allocated_chunks;

  // outstanding mlu notifiers
  std::deque<std::pair<std::shared_ptr<Notifier>, Chunk*>> notifiers;

  int count = 0;

  CachingAllocator()
      : large_chunks(ChunkComparator), small_chunks(ChunkComparator) {
    count = 0;
  }

  ~CachingAllocator() {
  }

  virtual MemoryStats &get_memory_stats_for_device(int device) {
    auto dev_count = device_count();
    auto cur_device = current_device();
    device = device == -1 ? cur_device : device;
    if (device >=0 && device < dev_count) {
      if ((size_t) device >= memory_stats.size()) {
        memory_stats.resize(device + 1);
      }
      return memory_stats.at(device);
    } else {
      LOG(FATAL) << "Caching Allocator: wrong device!";
    }
  }

  void insert_notifier(Chunk* chunk) {
    queue_set queues(std::move(chunk->queue_in_use));
    AT_ASSERT(chunk->queue_in_use.empty());
    for (auto it = queues.begin(); it != queues.end(); ++it) {
      c10::DeviceIndex device_id = static_cast<c10::DeviceIndex>(it->device_index());
      auto notifier_sptr = NotifierPool_Manager.alloc_notifier(device_id);
      notifier_sptr->place(*it);
      chunk->notifier_count++;
      notifiers.emplace_back(notifier_sptr, chunk);
    }
  }

  void process_notifiers() {
    while (!notifiers.empty()) {
      auto& n = notifiers.front();
      auto notifier_sptr = n.first;
      Chunk* chunk = n.second;
      torch_mlu::mlu::MLUGuard guard(notifier_sptr->device_index());
      const bool ret = notifier_sptr->query();
      if (ret == false) {
        break;
      }
      NotifierPool_Manager.give_back_notifier(notifier_sptr);
      chunk->notifier_count--;
      if (chunk->notifier_count == 0) {
        giveBackChunk(chunk);
      }
      notifiers.pop_front();
    }
  }

  void synchronize_and_free_notifier() {
    for (auto& n : notifiers) {
      auto notifier_sptr = n.first;
      Chunk* chunk = n.second;
      notifier_sptr->synchronize();
      NotifierPool_Manager.give_back_notifier(notifier_sptr);
      chunk->notifier_count--;
      if (chunk->notifier_count == 0) {
        giveBackChunk(chunk);
      }
      notifiers.pop_front();
    }
  }

  virtual void carveMasks(Chunk* chunk, Chunk* remain_chunk) {}

  virtual void carveHeader(Chunk* chunk) {}

  virtual void carveFooter(Chunk* chunk) {}

  virtual bool checkMask(Chunk* chunk) { return true;}

  virtual void recordBacktrace(Chunk* chunk) {}

  void malloc(void** data_ptr, size_t size, cnrtQueue_t queue, int device_id) {
    std::lock_guard<std::recursive_mutex> lock(base_mutex);

    process_notifiers();

    size = roundUpSize(size);

    auto &stats = get_memory_stats_for_device(device_id);

    Chunk searchChunk(device_id, queue, size);
    auto& pool = getChunkPool(size);

    auto findFreeChunk = [&]() -> Chunk* {
      auto it = pool.lower_bound(&searchChunk);
      if (it != pool.end() && (*it)->device_id == device_id &&
          (*it)->queue == queue) {
        Chunk* chunk = *it;
        pool.erase(it);
        return chunk;
      }
      return nullptr;
    };

    Chunk* chunk = findFreeChunk();
    if (chunk == nullptr) {
      void* ptr;
      size_t allocation_size = getAllocationSize(size);
      mluMalloc(device_id, &ptr, allocation_size, queue);
      stats.cached(allocation_size);
      chunk = new Chunk(device_id, queue, allocation_size, &pool, ptr);
      // DEBUG MODE: carve header and footer on chunk
      carveHeader(chunk);
      carveFooter(chunk);
    }

    Chunk* remain_chunk = nullptr;
    if (shouldSplit(chunk, size)) {
      remain_chunk = chunk;

      chunk = new Chunk(device_id, queue, size, &pool, chunk->ptr);
      chunk->prev = remain_chunk->prev;
      if (chunk->prev) {
        chunk->prev->next = chunk;
      }
      chunk->next = remain_chunk;

      remain_chunk->prev = chunk;
      remain_chunk->ptr = static_cast<char*>(remain_chunk->ptr) + size;
      remain_chunk->size -= size;
      carveMasks(chunk, remain_chunk);
      pool.insert(remain_chunk);
    }

    chunk->allocated = true;
    allocated_chunks[chunk->ptr] = chunk;

    *data_ptr = chunk->ptr;

    stats.allocated(chunk->size);

    recordBacktrace(chunk);

    #if USE_PROFILE
    profiler::reportMluMemoryUsageToProfiler(
      chunk, chunk->size, c10::Device(c10::DeviceType::MLU, device_id));
    #endif
  }

  void free(void* ptr) {
    std::lock_guard<std::recursive_mutex> lock(base_mutex);
    if (!ptr) {
      return;
    }

    auto it = allocated_chunks.find(ptr);
    if (it == allocated_chunks.end()) {
      AT_ERROR("invalid device pointer: ", ptr);
    }

    Chunk* chunk = it->second;
    allocated_chunks.erase(it);
    chunk->allocated = false;

    get_memory_stats_for_device(chunk->device_id).deallocated(chunk->size);

    #if USE_PROFILE
    profiler::reportMluMemoryUsageToProfiler(
      chunk, -chunk->size, c10::Device(c10::DeviceType::MLU, chunk->device_id));
    #endif

    if (!chunk->queue_in_use.empty()) {
      insert_notifier(chunk);
    } else {
      giveBackChunk(chunk);
    }
  }

  // moves a chunk into a pool of cached free chunks
  void giveBackChunk(Chunk* chunk) {
    checkMask(chunk);
    AT_ASSERT(!chunk->allocated && chunk->notifier_count == 0);
    auto& pool = *chunk->pool;
    mergeChunks(chunk, chunk->prev, pool);
    mergeChunks(chunk, chunk->next, pool);
    pool.insert(chunk);
  }

  // combine previously split chunks
  void mergeChunks(Chunk* dst, Chunk* src, ChunkPool& pool) {
    if (!src || src->allocated || src->notifier_count > 0) {
      return;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    dst->size += src->size;
    pool.erase(src);
    delete src;
  }

  // get chunk pool
  ChunkPool& getChunkPool(size_t size) {
    if (size <= small_allocation_size) {
      return small_chunks;
    } else {
      return large_chunks;
    }
  }

  virtual size_t roundUpSize(size_t size) {
    if (size < minimum_round_size) {
      return minimum_round_size;
    } else {
      return minimum_round_size *
             ((size + minimum_round_size - 1) / minimum_round_size);
    }
  }

  // get allocation size
  size_t getAllocationSize(size_t size) {
    auto native_memory_strategy = get_memory_strategy();
    size_t free = 0;
    size_t total = 0;
    // get free memory size(MiB)
    TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
    free = free / 1024 / 1024;
    total = total / 1024 / 1024;
    // get a quarter of free memory size(byte)
    size_t malloc_size = 0;
    if (native_memory_strategy) {
      malloc_size = size;
    } else {
      free = free * 1024 * 256;
      malloc_size = std::max(size, free);
    }
    if (size <= small_allocation_size) {
      return small_buffer_size;
    } else {
      if (malloc_size < large_allocation_size) {
        return large_buffer_size;
      } else {
        return maximum_round_size *
               ((malloc_size + maximum_round_size - 1) / maximum_round_size);
      }
    }
  }

  bool shouldSplit(Chunk* chunk, size_t size) {
    size_t remaining = chunk->size - size;
    if (chunk->pool == &small_chunks) {
      return remaining >= minimum_round_size;
    } else if (chunk->pool == &large_chunks) {
      return remaining > small_allocation_size;
    } else {
      AT_ERROR("shouldSplit: invalid MLU chunk pool");
    }
  }

  void mluMalloc(int device, void** data_ptr, size_t size, cnrtQueue_t& queue) {
    // first using cnrtMalloc, if fails then free all cached chunks and remalloc
    cnrtRet_t status = cnrtMalloc(data_ptr, size);
    if (status != cnrtSuccess) {
      free_cached_chunk(device);
      auto err = cnrtMalloc(data_ptr, size);
      if (err != cnrtSuccess) {
        if ((err == cnrtErrorNoMem) || (err == cnrtErrorCndrvFuncCall)) {
          size_t device_free;
          size_t device_total;
          cnrtChannelType_t channel = CNRT_CHANNEL_TYPE_DUPLICATE;
          TORCH_CNRT_CHECK(cnrtGetMemInfo(&device_free, &device_total, channel));
          const auto& stats = get_memory_stats_for_device(device);
          AT_ERROR(
            "CNRT out of memory. Tried to allocate ", format_size(size),
            " (MLU ", device, "; ",
            format_size(device_total * 1024 * 1024), " total capacity; ",
            format_size(stats.allocated_size), " already allocated; ",
            format_size(device_free * 1024 * 1024), " free; ",
            format_size(stats.cached_size - stats.allocated_size), " cached)");
        } else {
          TORCH_CNRT_CHECK(err);
        }
      }
    }
  }

  void free_cached_chunk(int device) {
    synchronize_and_free_notifier();

    Chunk lower_bound(device, nullptr, 0);
    Chunk upper_bound(device + 1, nullptr, 0);

    freeChunks(large_chunks, large_chunks.lower_bound(&lower_bound),
               large_chunks.lower_bound(&upper_bound));
    freeChunks(small_chunks, small_chunks.lower_bound(&lower_bound),
               small_chunks.lower_bound(&upper_bound));
  }

  void freeChunks(ChunkPool& chunks, ChunkPool::iterator it,
                  ChunkPool::iterator end) {
    std::lock_guard<std::mutex> lock(mlu_mutex);
    while (it != end) {
      Chunk* chunk = *it;
      if (!chunk->prev && !chunk->next) {
        cnrtRet_t status = cnrtFree((void*)chunk->ptr);
        if (status != CNRT_RET_SUCCESS) {
          // TODO(liuyuxin): free all cached chunks
          LOG(FATAL) << "Caching Allocator: cnrt free failed!";
        }
        get_memory_stats_for_device(chunk->device_id).decached(chunk->size);
        auto cur = it;
        ++it;
        chunks.erase(cur);
        delete chunk;
      } else {
        ++it;
      }
    }
  }

  void recordQueue(const c10::DataPtr& data_ptr, Queue queue) {
    auto ptr = data_ptr.get();
    if (!ptr) {
      return;
    }

    std::lock_guard<std::recursive_mutex> lock(std::mutex);

    if (debug_mode) {
      ptr = static_cast<char*>(ptr) - mask_bytes;
    }
    auto it = allocated_chunks.find(ptr);
    if (it == allocated_chunks.end()) {
      AT_ERROR("invalid device pointer, No allocated chunk can be found: ", ptr);
    }

    Chunk* chunk = it->second;
    if (queue.queue() == chunk->queue) {
        return;
    }
    chunk->queue_in_use.insert(queue);
  }

  void emptyCached() {
    std::lock_guard<std::recursive_mutex> lock(base_mutex);
    synchronize_and_free_notifier();
    freeChunks(large_chunks, large_chunks.begin(), large_chunks.end());
    freeChunks(small_chunks, small_chunks.begin(), small_chunks.end());
  }
};

// allocator using for memory management
CachingAllocator caching_allocator;

struct DebugAllocator : public CachingAllocator {
  // DEBUG MODE: Memory statistics
  std::vector<DebugStats> debug_memory_stats;

  // DEBUG MODE: saved backtrace;
  std::unordered_map<Chunk*, std::pair<int, char**>> saved_backtrace;

  DebugStats &get_memory_stats_for_device(int device) override {
    auto dev_count = device_count();
    auto cur_device = current_device();
    device = device == -1 ? cur_device : device;
    if (device >=0 && device < dev_count) {
      if ((size_t) device >= debug_memory_stats.size()) {
        debug_memory_stats.resize(device + 1);
      }
      return debug_memory_stats.at(device);
    } else {
      LOG(FATAL) << "Debug Allocator: wrong device!";
    }
  }

  void recordBacktrace(Chunk* chunk) override {
    void *buffer[layer_num];
    char **backtrace_str = nullptr;
    int size = backtrace(buffer, layer_num);
    backtrace_str = backtrace_symbols(buffer, size);
    if (backtrace_str) {
      saved_backtrace[chunk] = std::make_pair(size, backtrace_str);
    } else {
      return;
    }
  }

  void dumpBacktrace(Chunk* chunk) {
    auto iter = saved_backtrace.find(chunk);
    if (iter != saved_backtrace.end()) {
      int layer_num = iter->second.first;
      char** backtrace_str  = iter->second.second;
      for (int i = 0; i < layer_num; i++) {
        std::string sys_str(backtrace_str[i]);
        size_t start = sys_str.find('(') + 1;
        size_t end = sys_str.find('+');
        std::string sub = sys_str.substr(start, end - start);

        int status;
        const char* func_name = abi::__cxa_demangle(sub.c_str(), nullptr, nullptr, &status);
        LOG(INFO) << "stack[" << i << "] : " << func_name;
      }
    }
  }

  void carveHeader(Chunk* chunk) override {
    void* ptr = chunk->ptr;
    auto& queue = chunk->queue;
    auto size = chunk->size;
    TORCH_CNRT_CHECK(cnrtMemcpyAsync(ptr, header_mask.get(),
          mask_bytes, queue, CNRT_MEM_TRANS_DIR_HOST2DEV));
    TORCH_CNRT_CHECK(cnrtQueueSync(queue));
  }

  void carveFooter(Chunk* chunk) override {
    void* ptr = chunk->ptr;
    auto& queue = chunk->queue;
    auto size = chunk->size;
    TORCH_CNRT_CHECK(cnrtMemcpyAsync(static_cast<char*>(ptr) + (size - mask_bytes),
          footer_mask.get(), mask_bytes, queue, CNRT_MEM_TRANS_DIR_HOST2DEV));
    TORCH_CNRT_CHECK(cnrtQueueSync(queue));
  }

  // carve masks on the memory
  void carveMasks(Chunk* chunk, Chunk* remain_chunk) override {
      carveHeader(remain_chunk);
      carveFooter(chunk);
  }

  // round up masked size
  size_t roundUpSize(size_t size) override {
    size += 2 * mask_bytes;
    if (size < minimum_round_size) {
      return minimum_round_size;
    } else {
      return minimum_round_size * ((size + minimum_round_size - 1) / minimum_round_size);
    }
  }

  // check the ptr if in the chunks map
  void checkChunks(void* ptr) {
    char* chunk_ptr = reinterpret_cast<char*>(ptr) - mask_bytes;
    auto it = allocated_chunks.find(reinterpret_cast<void*>(chunk_ptr));
    if (it == allocated_chunks.end()) {
      throw ManageException();
    }
  }

  template<class T>
  Chunk* getC(T c) {
    return c;
  }


  // check mask of a chunk
  bool checkMask(Chunk* chunk) override {
    int64_t header[MASK_WORDS];
    int64_t footer[MASK_WORDS];
    bool no_error = true;
    void* ptr = chunk->ptr;
    size_t size = chunk->size;
    auto& queue = chunk->queue;
    TORCH_CNRT_CHECK(cnrtMemcpyAsync(header, ptr, mask_bytes, queue, CNRT_MEM_TRANS_DIR_DEV2HOST));
    TORCH_CNRT_CHECK(cnrtMemcpyAsync(footer, static_cast<char*>(ptr) + (size - mask_bytes),
          mask_bytes, queue, CNRT_MEM_TRANS_DIR_DEV2HOST));
    TORCH_CNRT_CHECK(cnrtQueueSync(queue));
    for (int i = 0; i < MASK_WORDS; ++i) {
      no_error &= (header[i] == header_mask.get()[i]);
      no_error &= (footer[i] == footer_mask.get()[i]);
      if (!no_error) {
        LOG(INFO) << "The memory is out of bound ! mask index = " << i
                   << " ;\n origin header mask = " << header_mask.get()[i]
                   << " , now header mask = " << header[i]
                   << " ;\n origin footer mask = " << footer_mask.get()[i]
                   << " , now footer mask = " << footer[i] << std::endl;
        dumpBacktrace(chunk);
      }
    }
    return no_error;
  }

  template<class T>
  bool checkPoolMask(T pool) {
    int64_t header[MASK_WORDS];
    int64_t footer[MASK_WORDS];
    bool no_error = true;
    for (auto c : pool) {
      // c is the copy of iterated elments of pool
      Chunk* chunk = nullptr;
      chunk = DebugAllocator::getC<decltype(c)>(c);
      no_error &= checkMask(chunk);
    }
    return no_error;
  }

  void checkMasks() {
    bool no_error = true;
    no_error &= checkPoolMask<decltype(allocated_chunks)>(allocated_chunks);
    no_error &= checkPoolMask<decltype(large_chunks)>(large_chunks);
    no_error &= checkPoolMask<decltype(small_chunks)>(small_chunks);
    if (!no_error) {
      throw  BoundException();
    }
  }
};

typedef std::pair<void* const, torch_mlu::Chunk*> Ck;
template<>
Chunk* DebugAllocator::getC<Ck>(Ck c) {
  return c.second;
}

// allocator using for memory debugging
DebugAllocator debugging_allocator;

// The library provides a recordQueue() function to help insert the correct
// synchronization when allocations are used on multiple queues. This will
// ensure that the chunk is not reused before each recorded queue completes
// work.
void recordQueue(const c10::DataPtr& data_ptr, Queue queue) {
  if (debug_mode) {
    debugging_allocator.recordQueue(data_ptr, queue);
  } else {
    caching_allocator.recordQueue(data_ptr, queue);
  }
}

inline void retriveDebugFlag() {
  char* env = std::getenv("ENABLE_CATCH_MEMORY_DEBUG");
  if (env != NULL) {
    debug_mode = (*env == '1');
  } else {
    debug_mode = false;
  }
}

static void MLUCachingDeleter(void* ptr) {
  if ( ptr == nullptr) return;
  retriveDebugFlag();
  if (debug_mode) {
    ptr = static_cast<char*>(ptr) - mask_bytes;
    debugging_allocator.free(ptr);
  } else {
    caching_allocator.free(ptr);
  }
}

c10::DataPtr MLUCachingAllocator::allocate(size_t size) const {
  // fake allocation, only for setting device
  auto device_id = current_device();
  void* data = nullptr;
  return {data, data, &MLUCachingDeleter,
          c10::Device(c10::DeviceType::MLU, static_cast<int16_t>(device_id))};
}

c10::DataPtr MLUCachingAllocator::allocate(size_t size,
                                           c10::DeviceIndex device_id) const {
  void* data = nullptr;
  retriveDebugFlag();
  if (debug_mode) {
    debugging_allocator.malloc(&data, size, getCurQueue(device_id), static_cast<int>(device_id));
    data = static_cast<char*>(data) + mask_bytes;
  } else {
    caching_allocator.malloc(&data, size, getCurQueue(device_id), static_cast<int>(device_id));
  }
  return {data, data, &MLUCachingDeleter,
          c10::Device(c10::DeviceType::MLU, static_cast<int16_t>(device_id))};
}

c10::DeleterFnPtr MLUCachingAllocator::raw_deleter() const {
  return &MLUCachingDeleter;
}

// using in at::empty
MLUCachingAllocator mlu_caching_allocator;

C10_API c10::Allocator* getMLUCachingAllocator(void) {
  return &mlu_caching_allocator;
}

// return the current memory allocated on MLU
uint64_t currentMemoryAllocated(int device_id) {
  retriveDebugFlag();
  if (debug_mode) {
    return debugging_allocator.get_memory_stats_for_device(device_id).allocated_size;
  } else {
    return caching_allocator.get_memory_stats_for_device(device_id).allocated_size;
  }
}

// return the current memory cached on MLU
uint64_t currentMemoryCached(int device_id) {
  retriveDebugFlag();
  if (debug_mode) {
    return debugging_allocator.get_memory_stats_for_device(device_id).cached_size;
  } else {
    return caching_allocator.get_memory_stats_for_device(device_id).cached_size;
  }
}

// return the max memory allocated on MLU
uint64_t maxMemoryAllocated(int device_id) {
  retriveDebugFlag();
  if (debug_mode) {
    return debugging_allocator.get_memory_stats_for_device(device_id).max_allocated_size;
  } else {
    return caching_allocator.get_memory_stats_for_device(device_id).max_allocated_size;
  }
}

// return the max memory cached on MLU
uint64_t maxMemoryCached(int device_id) {
  retriveDebugFlag();
  if (debug_mode) {
    return debugging_allocator.get_memory_stats_for_device(device_id).max_cached_size;
  } else {
    return caching_allocator.get_memory_stats_for_device(device_id).max_cached_size;
  }
}

// empty all cached and unchained memory
void emptyCachedMem() {
  retriveDebugFlag();
  if (debug_mode) {
    return debugging_allocator.emptyCached();
  } else {
    return caching_allocator.emptyCached();
  }
}

// set debug env value (only for gtest)
void setDebugEnv(char* flag) {
  char* env = std::getenv("ENABLE_CATCH_MEMORY_DEBUG");
  int overwrite = 0;
  if (env == NULL) {
    overwrite = 1;
  } else {
    overwrite = (*env == *flag) ? 0 : 1;
  }
  int status = setenv("ENABLE_CATCH_MEMORY_DEBUG", flag , overwrite);
  if (status != 0) {
      AT_ERROR("set env value failed : ENABLE_CATCH_MEMORY_DEBUG");
  }
}

// memory debugging
void memoryDebug(c10::DataPtr* data) {
  if (data->device().type() != c10::DeviceType::MLU) {
    LOG(INFO) << "storage of non-MLU type can not debugged by allocator!!";
    return;
  }
  LOG(INFO) << "===================== Checking Memory Out of Bound ...  =====================";
  debugging_allocator.checkMasks();
  LOG(INFO) << "===================== No Memory Out of Bound !!! =====================";
  debugging_allocator.checkChunks(data->get());
  LOG(INFO) << "===================== Storage is managed by allocator !!! =====================";
}

void memoryDebug(const c10::DataPtr* data) {
  memoryDebug(const_cast<c10::DataPtr*>(data));
}

void memoryDebug() {
  LOG(INFO) << "===================== Checking Memory Out of Bound ...  =====================";
  debugging_allocator.checkMasks();
}
}  // namespace torch_mlu
