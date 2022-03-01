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

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>

#include "aten/core/tensor_impl.h"
#include "aten/device/mlu_guard.h"
#include "aten/device/queue.h"
#include "aten/util/matrix_util.h"
#include "aten/util/memory_allocator.h"

namespace torch_mlu {

// MLUTensorInfo end
//
MLUTensorImpl::MLUTensorImpl(c10::Storage&& storage, c10::DispatchKey type_id,
                             const caffe2::TypeMeta& data_type)
    : c10::TensorImpl(std::move(storage), type_id, data_type) {
  cpu_storage_ =
      c10::Storage::create_legacy(c10::DeviceType::CPU);
  cnnl_data_type_ = getCnnlDataType(data_type_);
}

MLUTensorImpl::MLUTensorImpl(c10::Storage&& storage,
                             c10::DispatchKeySet type_set,
                             const caffe2::TypeMeta& data_type)
    : c10::TensorImpl(std::move(storage), type_set, data_type) {
  cpu_storage_ =
      c10::Storage::create_legacy(c10::DeviceType::CPU);
  cnnl_data_type_ = getCnnlDataType(data_type_);
}

MLUTensorImpl::~MLUTensorImpl() {}

// the logic of this function comes from MLUTensorImpl::cpu_data(), although
// not so faithfully. by the time of writing, it is used exclusively by the
// fallback-to-cpu feature, to denote the case where empty_like() should replace
// copy_()
// if is_empty() were to be applied alsewhere, the logic may need to be refined,
// but now we will leave it as is because when entering this function from
// inputConvertAndPushTensor(), the tensor is already verified as 'defined'
bool MLUTensorImpl::is_empty() {
  if (!cpu_storage_ || !storage_)
    return true;

  if (storage_.data() == nullptr && cpu_storage_.data() == nullptr)
    return true;

  return false;
}

void* MLUTensorImpl::cpu_data() {
  if (!cpu_storage_ || !storage_) {
    std::string warning = "It should exist cpu_storage_ and storage_ ";
    warning += "while calling this function. Please report this error to us.";
    throw std::invalid_argument(warning);
  }
  // For a MLU tensor, we use cpu_storage_ to store CPU data and storage_ for
  // MLU data. For tensor without cpu_storage_, copy storage from mlu to cpu.
  at::DataPtr cpu_data_ptr;
  at::DataPtr cpu_data_ptr_cast;
  if (storage_.data() != nullptr) {
    has_already_copy = true;
    return cpu_storage_.data();
  } else if (cpu_storage_.data() != nullptr) {
    return static_cast<void*>(static_cast<char*>(this->cpu_storage_.data()) +
                              data_type_.itemsize() * storage_offset_);
  } else {
    // both cpu_storage and mlu_storage are not initialized
    CNLOG(ERROR) << "\n\nBoth cpu_storage and mlu_storage are not initialized!\n"
                 << "Please check is there any invalid tensor operates such as:\n"
                 << "output = input.cpu() or output = input.to(\"cpu\") "
                 << "in pytorch model when doing mlu/mfus inference.\n";
    AT_ERROR("Can not call cpu_data on an empty tensor.");
    return nullptr;
  }
}

void MLUTensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

const at::Storage& MLUTensorImpl::cpu_storage() const { return cpu_storage_; }

void MLUTensorImpl::set_cpu_storage(at::Storage storage) {
  cpu_storage_ = std::move(storage);
}

void MLUTensorImpl::set_cpu_storage_offset(int64_t storage_offset) {
  cpu_storage_offset_ = storage_offset;
}


c10::intrusive_ptr<c10::TensorImpl> MLUTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<MLUTensorImpl>(c10::Storage(this->storage()),
                                                 this->key_set_,
                                                 this->data_type_);

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
  impl->refresh_contiguous();

  return impl;
}

void MLUTensorImpl::set_has_already_copy_flag(bool flag) {
  has_already_copy = flag;
}

void MLUTensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl, TensorImpl* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  // Call the copy_tensor_metadata() of base Class
  c10::TensorImpl::copy_tensor_metadata(src_impl, dest_impl, version_counter,
                                        allow_tensor_metadata_change);

  /**
   *One TensorImpl can be copied to another TensorImpl if they have the same DispatchKeySet.
   * Support a special case: CPU is compatible with MLU.
   */
  if (src_impl->device_type() == c10::DeviceType::CPU) {
      return;
  }
  // Copy the tensor metadata fields which are unique for MLU
  auto src_mlu_impl =
      dynamic_cast<MLUTensorImpl*>(const_cast<c10::TensorImpl*>(src_impl));
  auto dest_mlu_impl = dynamic_cast<MLUTensorImpl*>(dest_impl);

  dest_mlu_impl->cpu_storage_ = src_mlu_impl->cpu_storage_;
  dest_mlu_impl->cpu_storage_offset_ = src_mlu_impl->cpu_storage_offset_;
  dest_mlu_impl->cnnl_data_type_ = src_mlu_impl->cnnl_data_type_;
  dest_mlu_impl->shared_op_info_ = src_mlu_impl->shared_op_info_;
  dest_mlu_impl->graph_seg_record_ = src_mlu_impl->graph_seg_record_;
  dest_mlu_impl->using_shared_op_info_ = src_mlu_impl->using_shared_op_info_;
}

void MLUTensorImpl::update_cpu_storage(void* cpu_data_ptr) {
  auto new_cpu_storage =
      c10::Storage::create_legacy(c10::DeviceType::CPU);
  int16_t device_id = static_cast<int16_t>(cpu_storage_.device().index());
  c10::DataPtr data_ptr = {cpu_data_ptr,
                           c10::Device(c10::DeviceType::MLU, device_id)};
  new_cpu_storage.set_data_ptr(std::move(data_ptr));
  this->set_cpu_storage(new_cpu_storage);
}


void MLUTensorImpl::copy_cnnl_metadata_from(const MLUTensorImpl* src_mlu_impl) {
  // set_data_ptr + std::move may cause unintentional release of DataPtr when
  // src and dst Tensor share a same storage
  if (!(storage_.is_alias_of(src_mlu_impl->storage_))) {
    storage_.set_data_ptr(
        std::move(const_cast<MLUTensorImpl*>(src_mlu_impl)->storage_.data_ptr()));
    storage_.set_nbytes(const_cast<MLUTensorImpl*>(src_mlu_impl)->storage_.nbytes());
  }
  cnnl_data_type_ = src_mlu_impl->cnnl_data_type_;
  shared_op_info_ = src_mlu_impl->shared_op_info_;
  graph_seg_record_ = src_mlu_impl->graph_seg_record_;
  using_shared_op_info_ = src_mlu_impl->using_shared_op_info_;
}

void MLUTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<c10::TensorImpl>& impl) {
  copy_tensor_metadata(
      /*src_impl=*/impl.get(),
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  refresh_numel();
  refresh_contiguous();
}


// for cnnl malloc data
void* MLUTensorImpl::cnnlMalloc(bool non_blocking) {
  // TODO(zhujing): to add check for mlu
  // AT_CHECK(is_mlu(), "cnnlMalloc only surpport mlu Tensor!");
  if (numel() == 0) return nullptr;
  cnrtDataType_t cnrt_dtype_onchip = CnnlType2CnrtType(getCnnlType());
  size_t itemsize_onchip = cnrtDataTypeSize(cnrt_dtype_onchip);
  size_t mlu_tensor_size = storage_.nbytes() / itemsize() * itemsize_onchip;


  if (storage_.data() == nullptr) {
    torch_mlu::mlu::MLUGuard guard(this->get_device());
    const torch_mlu::MLUCachingAllocator* allocator =
        dynamic_cast<torch_mlu::MLUCachingAllocator*>(
            this->storage_.allocator());
    if (allocator == nullptr) {
      allocator = dynamic_cast<torch_mlu::MLUCachingAllocator*>(
          at::GetAllocator(this->storage_.device_type()));
    }
    auto device_id = storage_.device().index();
    storage_.set_data_ptr(allocator->allocate(mlu_tensor_size, device_id));
  }
  if (cpu_storage_.data() && !has_already_copy) {
    auto cpu_memory_offset = itemsize() * cpu_storage_offset_;
    TORCH_MLU_CHECK(storage_.nbytes() <= (cpu_storage_.nbytes() - cpu_memory_offset),
                "cpu2mlu copy in cnnlMalloc failed! CPU memory is smaller than MLU");
    /**** data cast ****/
    void* cpu_ptr =
        static_cast<void*>(static_cast<char*>(cpu_storage_.data()) + cpu_memory_offset);
    void* cast_cpu_ptr = cpu_ptr;
    cnrtDataType_t cnrt_dtype_offchip = toCnrtDtype(data_type_);
    // currently do not support 64bit data on chip, so cnrt_dtype_offchip may not
    // eq cnrt_dtype_onchip
    bool is_pinned = isPinned<void>(cast_cpu_ptr);
    auto* pinned_allocator = getMLUCachingHostAllocator();
    if (cnrt_dtype_offchip == CNRT_INT64) {
      if (non_blocking) {
        cast_cpu_ptr = pinned_allocator->raw_allocate(mlu_tensor_size);
        is_pinned = true;
      } else {
        cast_cpu_ptr = allocateGlobalBuffer<void>(mlu_tensor_size);
      }
      if (cnrt_dtype_onchip == CNRT_INT16) {
        cast_cpu_op<long, short>(cast_cpu_ptr, cpu_ptr);
      } else {
        cast_cpu_op<long, int>(cast_cpu_ptr, cpu_ptr);
      }
    } else if (cnrt_dtype_offchip == CNRT_FLOAT64) {
      if (non_blocking) {
        cast_cpu_ptr = pinned_allocator->raw_allocate(mlu_tensor_size);
        is_pinned = true;
      } else {
        cast_cpu_ptr = allocateGlobalBuffer<void>(mlu_tensor_size);
      }
      CNLOG(INFO) << "cnnl not support real double computation," <<
                     " and cast to float computation instead";
      cast_cpu_op<double, float>(cast_cpu_ptr, cpu_ptr);
    }
    /**** cpu2mlu copy ****/
    auto queue = getCurrentQueue();
    CNRT_CHECK(cnrtMemcpyAsync(storage_.data(), cast_cpu_ptr,
                               mlu_tensor_size, queue.queue(),
                               CNRT_MEM_TRANS_DIR_HOST2DEV));
    if (cnrt_dtype_offchip == CNRT_INT64 ||
        cnrt_dtype_offchip == CNRT_FLOAT64) {
      if (non_blocking) {
        pinned_allocator->raw_deleter()(cast_cpu_ptr);
      } else {
      GlobalBuffer_placeNotifier(cast_cpu_ptr);
      }
    }
    if (!(non_blocking && is_pinned)) {
      queue.synchronize();
    }
    has_already_copy = true;  // cpu2mlu
  }
  return static_cast<void*>(static_cast<char*>(storage_.data()) +
                            itemsize_onchip * storage_offset_);
}

// compare input tensor and last node output tensor in graph.
// compare item include: size/stride/sotrage_offset.
bool MLUTensorImpl::is_compare_size_stride_info(const MLUTensorImpl*& lhs) {
  return this->sizes() == lhs->sizes() && this->strides() == lhs->strides()
    && this->storage_offset() == lhs->storage_offset();
}

bool MLUTensorImpl::is_compare_size_stride_info(const TupleRecordOpInfo& other) {
  return this->sizes().vec() == std::get<0>(other)
         && this->strides().vec() == std::get<1>(other)
         && this->storage_offset() == std::get<2>(other);
}

// inplace op using this interface, insert views op node.
void MLUTensorImpl::insert_views_op_info(const VIEWOPNAME& name,
            const typename MLUTensorImpl::TupleRecordOpInfo& input,
            const bool input_contiguous) {
  if (is_compare_size_stride_info(input) == true) {
    return;
  }
  if (using_shared_op_info_ == false) {
    graph_seg_record_.clear();
    shared_op_info_.clear();
  }

  typename MLUTensorImpl::TupleRecordOpInfo output = std::make_tuple(sizes().vec(),
                                                                     strides().vec(),
                                                                     storage_offset());
  auto op_info = std::make_tuple(name, input, output);
  // add new op to shared_ops
  check_add_op_into_shared_ops(op_info);
  auto it = std::find(shared_op_info_.begin(), shared_op_info_.end(), op_info);
  int op_index = std::distance(shared_op_info_.begin(), it);

  if (input_contiguous == true) {
    graph_seg_record_.push_back(std::vector<int>({op_index}));
  } else {
    // find start node in each graph, and copy index list.
    int index_of_graph = 0;
    int index_of_inner_graph = 0;
    std::tie(index_of_graph, index_of_inner_graph) = search_op_segm_index(input);
    if (index_of_graph == -1 && index_of_inner_graph == -1) {
      // can't find op info in graph, so add a graph seg to graph_seg_record_;
      // Specially this func is for non-contiguous input of graph.
      graph_seg_record_.push_back(std::vector<int>({op_index}));
      using_shared_op_info_ = true;
      return;
    }
    TORCH_CHECK(index_of_graph < graph_seg_record_.size(),
      "index out of graph_seg_record_ range.");
    TORCH_CHECK(index_of_inner_graph < graph_seg_record_[index_of_graph].size(),
      "index out of graph_seg_record_ inner graph seg range.");
    const int graph_len = graph_seg_record_[index_of_graph].size() - 1;
    if (index_of_inner_graph == graph_len) {
      // a new node push to old graph.
      graph_seg_record_[index_of_graph].push_back(op_index);
    } else {
      // copy a new graph for push op node
      std::vector<int> new_graph_seg(graph_seg_record_[index_of_graph].begin(),
        graph_seg_record_[index_of_graph].begin()+index_of_inner_graph);
      new_graph_seg.push_back(op_index);
      // if graph already in graph_seg_record_, pass;
      // else push to graph_seg_record_.
      check_and_add_new_graph(new_graph_seg);
    }
  }
  using_shared_op_info_ = true;
  }


// insert views op node.
void MLUTensorImpl::insert_views_op_info(const VIEWOPNAME& name,
                          const MLUTensorImpl* input_tensor_impl,
                          const bool input_contiguous) {
  graph_seg_record_ = input_tensor_impl->graph_seg_record_;
  shared_op_info_ = input_tensor_impl->shared_op_info_;
  using_shared_op_info_ = input_tensor_impl->using_shared_op_info_;

  typename MLUTensorImpl::TupleRecordOpInfo input = std::make_tuple(
                    input_tensor_impl->sizes().vec(),
                    input_tensor_impl->strides().vec(),
                    input_tensor_impl->storage_offset());
  insert_views_op_info(name, input, input_contiguous);
}

// return graph segment to do contiguous
std::vector<typename MLUTensorImpl::SharedOpInfo> MLUTensorImpl::get_views_op_info(
                        const typename MLUTensorImpl::TupleRecordOpInfo& input) {
  if (using_shared_op_info_ == false) {
    return std::vector<typename MLUTensorImpl::SharedOpInfo>();
  }
  const int graph_size_len = graph_seg_record_.size();
  const int shared_op_len = shared_op_info_.size();
  TORCH_CHECK(graph_size_len != 0,
      "graph_seg_record_ is null.");
  TORCH_CHECK(shared_op_len != 0,
      "shared_op_info_ is null.");
  int seg_index = 0;
  bool find_seg = false;
  for (int i = 0; i < graph_size_len; ++i) {
    const int index = graph_seg_record_[i].back();
    TORCH_CHECK(index < shared_op_len,
      "Index is bigger than shared_op_len.");
    if (std::get<2>(shared_op_info_[index]) == input) {
      seg_index = i;
      find_seg = true;
      break;
    }
  }
  if (find_seg == false) {
    return std::vector<typename MLUTensorImpl::SharedOpInfo>();
  }
  TORCH_CHECK(seg_index < graph_size_len,
      "Index out of graph_size_len range.");
  std::vector<typename MLUTensorImpl::SharedOpInfo> graph_seg;
  for (const auto& index : graph_seg_record_[seg_index]) {
    TORCH_CHECK(index < shared_op_len,
      "Index out of shared_op_len range.");
    graph_seg.push_back(shared_op_info_[index]);
  }
  return graph_seg;
}
}  // namespace torch_mlu
