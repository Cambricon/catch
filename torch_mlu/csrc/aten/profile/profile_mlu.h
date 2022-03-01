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

#pragma once

#include <cnlight.h>
#include <c10/core/Allocator.h>
#include <ATen/record_function.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/profiler.h>

#include <mutex>
#include <vector>
#include <string>
#include <cstdint>

namespace torch_mlu {
namespace profiler {

/*!
 * @class DeviceEventVisitor
 *
 * @brief Provide interface to obtain the detailed information of device events.
 *  DeviceEventVisitor can be used to provide Python apis.
 *
 * @par Requirements
 * - cnlight.h
 *
 * @note
 * - None
*/
class DeviceEventVisitor {
 public:
  explicit DeviceEventVisitor(const cnlight::trace::DeviceEvent& device_event):
      device_event_(device_event) {}

  /* TODO(hanfei): To use for python debug. */
  // const char* type() const {
  //   switch (device_event_.type_) {
  //     case cnlight::trace::DeviceEvent::Type::Kernel: return "kernel";
  //     case cnlight::trace::DeviceEvent::Type::MemcpyH2D: return "MemcpyH2D";
  //     case cnlight::trace::DeviceEvent::Type::MemcpyD2H: return "MemcpyD2H";
  //     case cnlight::trace::DeviceEvent::Type::MemcpyD2D: return "MemcpyD2D";
  //     case cnlight::trace::DeviceEvent::Type::MemorySet: return "MemorySet";
  //     default: return "Unknown";
  //   }
  // }

  const char* name() const { return device_event_.name_.c_str(); }

  int64_t start_ns() const { return device_event_.start_ns_; }

  int64_t duration_ns() const { return device_event_.duration_ns_; }

  /* TODO(hanfei): To use for python debug. */
  // const char* device_type() const {
  //   switch (device_event_.device_type_) {
  //     case cnlight::trace::DeviceType::CPU: return "CPU";
  //     case cnlight::trace::DeviceType::GPU: return "GPU";
  //     case cnlight::trace::DeviceType::MLU: return "MLU";
  //     default: return "UNSPECIFIED";
  //   }
  // }

  int64_t device_index() const { return device_event_.device_index_; }

  /* TODO(hanfei): To use for python debug. */
  // std::string toString() const {
  //   return "type:         " + std::string(type()) + "\n" +
  //          "name:         " + std::string(name()) + "\n" +
  //          "start_ns:     " + std::to_string(start_ns()) + "\n" +
  //          "duration_ns:  " + std::to_string(duration_ns()) + "\n" +
  //          "device_type:  " + std::string(device_type()) + "\n" +
  //          "device_index: " + std::to_string(device_index());
  // }

 private:
  const cnlight::trace::DeviceEvent& device_event_;
};


/*!
 * @class ElapseEventVisitor
 *
 * @brief Provide interface to obtain the detailed information of elapse events.
 *  ElapseEventVisitor can be used to provide Python apis.
 *
 * @par Requirements
 * - cnlight.h
 *
 * @note
 * - None
*/
class ElapseEventVisitor {
 public:
  explicit ElapseEventVisitor(const cnlight::trace::ElapseEvent& elapse_event):
    elapse_event_(elapse_event) {}

  int64_t id() const { return elapse_event_.id_; }

  /* TODO(hanfei): To use for python debug. */
  // int64_t correlation_id() const { return elapse_event_.correlation_id_; }

  // const char* name() const { return elapse_event_.name_.c_str(); }

  const char* type() const { return elapse_event_.type_.c_str(); }

  int64_t start_ns() const { return elapse_event_.start_ns_; }

  int64_t end_ns() const { return elapse_event_.end_ns_; }

  int64_t start_thread_id() const { return elapse_event_.start_thread_id_; }

  int64_t end_thread_id() const { return elapse_event_.end_thread_id_; }

  int64_t parent() const { return elapse_event_.parent_; }

  std::vector<int64_t> children() const { return elapse_event_.children_; }

  std::vector<DeviceEventVisitor> device_events() const {
    std::vector<DeviceEventVisitor> device_events_visitor;
    device_events_visitor.reserve(elapse_event_.device_events_.size());
    for (auto &device_event : elapse_event_.device_events_) {
      device_events_visitor.emplace_back(DeviceEventVisitor(device_event));
    }
    return std::move(device_events_visitor);
  }

  /* TODO(hanfei): To use for python debug. */
  // std::string toString() const {
  //   std::string out = "id:              " + std::to_string(id()) + "\n" +
  //                     "correlation_id:  " + std::to_string(correlation_id()) + "\n" +
  //                     "name:            " + std::string(name()) + "\n" +
  //                     "type:            " + std::string(type()) + "\n" +
  //                     "start_ns:        " + std::to_string(start_ns()) + "\n" +
  //                     "end_ns:          " + std::to_string(end_ns()) + "\n" +
  //                     "start_thread_id: " + std::to_string(start_thread_id()) + "\n" +
  //                     "parent:          " + std::to_string(parent()) + "\n" +
  //                     "children:        ";

  //   for (auto& child : children()) {
  //     out += std::to_string(child) + " ";
  //   }

  //   out += "\ndevice_events:\n";
  //   for (auto& device : device_events()) {
  //     out += device.toString();
  //     out += "\n";
  //   }
  //   return out;
  // }

 private:
  const cnlight::trace::ElapseEvent& elapse_event_;
};


/*!
 * @class ProfilerMemoryEvent
 *
 * @brief Record memory event.
 *
 * @note
 * - None
*/
struct ProfilerMemoryEvent final {
  void updateMemoryStats(int64_t alloc_size, c10::Device device) {
    if (device.type() == c10::DeviceType::MLU) {
      mlu_memory_usage_ += alloc_size;
    } else if (device.type() == c10::DeviceType::CPU ||
        device.type() == c10::DeviceType::MKLDNN ||
        device.type() == c10::DeviceType::IDEEP) {
      cpu_memory_usage_ += alloc_size;
    } else {
      LOG(WARNING) << "Unsupported memory profiling device: " << device;
    }
  }

  int64_t getCpuMemoryUsage() const {
    return cpu_memory_usage_;
  }

  int64_t getMluMemoryUsage() const {
    return mlu_memory_usage_;
  }

  int64_t cpu_memory_usage_ = 0;
  int64_t mlu_memory_usage_ = 0;
};


/*!
 * @class ProfilerMluContext
 *
 * @brief Record remain event except elapse_event.
 *
 * @note
 * - None
*/
struct ProfilerMluContext final {
  explicit ProfilerMluContext(
    std::vector<std::vector<int64_t>>&& shapes = {},
    at::RecordFunctionHandle handle = 0,
    int node_id = -1)
    : shapes_(shapes), handle_(handle), node_id_(node_id) {}

  ProfilerMemoryEvent& getMemoryEvent() { return memory_event_; }

  ProfilerMemoryEvent memory_event_;
  std::vector<std::vector<int64_t>> shapes_;
  at::RecordFunctionHandle handle_ {0};
  int node_id_ = 0;
};


/*!
 * @class ProfilerMluEvent
 *
 * @brief Record mlu events collecting information.
 *  ProfilerMluEvent can be used to provide Python apis.
 *
 * @par Requirements
 * - cnlight.h
 *
 * @note
 * - None
*/
struct ProfilerMluEvent final {
  ProfilerMluEvent(
    int64_t correlation_id,
    cnlight::trace::ElapseEvent&& elapse_event,
    std::vector<std::vector<int64_t>>&& shapes = {},
    at::RecordFunctionHandle handle = 0,
    int node_id = -1, ProfilerMemoryEvent memory_event = {0, 0})
    : correlation_id_(correlation_id),
    elapse_event_(elapse_event), shapes_(shapes),
    handle_(handle), node_id_(node_id),
    memory_event_(memory_event) {}

  ProfilerMluEvent(
    int64_t correlation_id,
    cnlight::trace::ElapseEvent&& elapse_event,
    ProfilerMluContext* mlu_context_ptr)
    : correlation_id_(correlation_id),
    elapse_event_(elapse_event), shapes_(mlu_context_ptr->shapes_),
    handle_(mlu_context_ptr->handle_), node_id_(mlu_context_ptr->node_id_),
    memory_event_(mlu_context_ptr->memory_event_) {}

  /* TODO(hanfei): To use for python debug. */
  // int64_t correlation_id() const {
  //   return correlation_id_;
  // }

  ElapseEventVisitor elapse_event() const {
    return ElapseEventVisitor(elapse_event_);
  }

  std::vector<std::vector<int64_t>> shapes() const {
    return shapes_;
  }

  at::RecordFunctionHandle handle() const {
    return handle_;
  }

  // Node ID corresponding to this event.
  int node_id() const {
    return node_id_;
  }

  int64_t cpu_memory_usage() const {
    return memory_event_.getCpuMemoryUsage();
  }

  int64_t mlu_memory_usage() const {
    return memory_event_.getMluMemoryUsage();
  }

  // std::string toString() const {
  //   std::string out = "correlation_id: " + std::to_string(correlation_id());

  //   out += "\nelapse_event:\n";
  //   out += elapse_event().toString();

  //   out += "\nhandle:         " + std::to_string(handle()) + "\n" +
  //            "node_id:        " + std::to_string(node_id()) + "\n";

  //   out += "cpu_memory:     " + std::to_string(cpu_memory_usage()) + "\n" +
  //          "mlu_memory:     " + std::to_string(mlu_memory_usage());
  //   return out;
  // }

 private:
  int64_t correlation_id_;
  cnlight::trace::ElapseEvent elapse_event_;
  ProfilerMemoryEvent memory_event_;
  std::vector<std::vector<int64_t>> shapes_;
  at::RecordFunctionHandle handle_ {0};
  int node_id_ = 0;
};


/*!
 * @class RangeProfilerMluEvent
 *
 * @brief To record ProfilerMluEvent to list.
 *  RangeProfilerMluEvent is used to merge collection information.
 *
 * @par
 *
 * @note
 * - None
*/
struct RangeProfilerMluEvent {
  RangeProfilerMluEvent() {
    events_.reserve(kReservedCapacity);
  }

  template<typename... Args>
  void record(Args&&... args) {
    std::lock_guard<std::mutex> guard(mutex_);
    events_.emplace_back(std::forward<Args>(args)...);
  }

  std::vector<ProfilerMluEvent> consolidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ProfilerMluEvent> result;
    result.insert(
        result.begin(),
        std::make_move_iterator(events_.begin()),
        std::make_move_iterator(events_.end()));
    events_.erase(events_.begin(), events_.end());
    return result;
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
  }

 private:
  // This mutex is used to serialize access when different threads are writing
  // to the same instance of RangeEventList.
  std::mutex mutex_;
  std::vector<ProfilerMluEvent> events_;

  static const size_t kReservedCapacity = 1024;
};

using thread_event_lists = std::vector<std::vector<ProfilerMluEvent>>;

TORCH_API void enableMluProfiler(const torch::autograd::profiler::ProfilerConfig& config);

TORCH_API thread_event_lists disableMluProfiler();

TORCH_API void reportMluMemoryUsageToProfiler(void* ptr, int64_t alloc_size, c10::Device device);
}  // namespace profiler
}  // namespace torch_mlu
