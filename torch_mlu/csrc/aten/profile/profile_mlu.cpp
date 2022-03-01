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

#include <profile_mlu.h>

#include <c10/util/ThreadLocalDebugInfo.h>

#include <atomic>
#include <memory>
#include <list>
#include <unordered_map>
#include <algorithm>

#include "aten/util/cnlog.h"

namespace torch_mlu { namespace profiler {

namespace {

uint64_t next_correlation_id() {
  static std::atomic<uint64_t> corr_id_ {1};
  return corr_id_++;
}

thread_local std::vector<std::shared_ptr<at::RecordFunctionGuard>> g_;

// TODO(hanfei): Temporary add codes to control hostlevel and devicelevel.
cnlight::trace::HostTracerLevel getHostTracerLevel(std::string host_level) {
  if (host_level == "1") {
    return cnlight::trace::HostTracerLevel::kCritical;
  } else if (host_level == "2") {
    return cnlight::trace::HostTracerLevel::kInfo;
  } else if (host_level == "3") {
    return cnlight::trace::HostTracerLevel::kVerbose;
  } else {
    CNLOG(WARNING) << "Host Trace turn off";
    return cnlight::trace::HostTracerLevel::kOff;
  }
}

cnlight::trace::DeviceTracerLevel getDeviceTracerLevel(std::string device_level) {
  if (device_level == "1") {
    return cnlight::trace::DeviceTracerLevel::kOn;
  } else {
    CNLOG(WARNING) << "Device Trace turn off";
    return cnlight::trace::DeviceTracerLevel::kOff;
  }
}

struct MLUThreadLocalState : public c10::MemoryReportingInfoBase {
  explicit MLUThreadLocalState(
      const torch::autograd::profiler::ProfilerConfig& config)
    : config_(config) {}

  inline const torch::autograd::profiler::ProfilerConfig& config() const {
    return config_;
  }

  at::CallbackHandle callbackHandle() const {
    return handle_;
  }

  void setCallbackHandle(at::CallbackHandle handle) {
    handle_ = handle;
  }

  void setTracer(cnlight::trace::TracerInterface* trace, const char* tracer_output) {
    tracer_ = trace;
    tracer_output_ = tracer_output;
  }

  cnlight::trace::TracerInterface* getTracer() {
    return tracer_;
  }

  std::string* getCollectionPtr() {
    return &collection_data_;
  }

  const char* getTracerOutPath() {
    return tracer_output_;
  }

  void addProfilerMluContext(const int64_t correlation_id,
                             const ProfilerMluContext& mlu_context) {
    context_map_[correlation_id] = mlu_context;
  }

  void pushContext(
      const at::StringView& name,
      const std::string& correlation_id_ptr,
      std::vector<std::vector<int64_t>>&& shapes = {},
      at::RecordFunctionHandle handle = 0,
      int node_id = -1) {
    std::lock_guard<std::mutex> g(state_mutex_);
    ProfilerMluContext mlu_context(std::move(shapes), handle, node_id);

    int64_t correlation_id = static_cast<int64_t> (std::stoll(correlation_id_ptr));
    addProfilerMluContext(correlation_id, mlu_context);

    auto trace = std::make_shared<cnlight::trace::AnnotatedTraceMe> (
      [&] { return cnlight::trace::TraceMeEncode(
          std::string(name.str()) + correlation_id_ptr +
                                ":" + std::string(name.str()),
                          {{"traceme_correlation_id", correlation_id_ptr}});});
    tracemes.push_front(std::make_tuple(name, correlation_id, trace));
  }

  void popContext(const at::StringView& name) {
    std::lock_guard<std::mutex> g(state_mutex_);
    TORCH_CHECK(!tracemes.empty(), "POP Context Error");

    auto tracemes_iter = find_if(tracemes.begin(), tracemes.end(),
        [&name](std::tuple<at::StringView, int64_t,
          std::shared_ptr<cnlight::trace::AnnotatedTraceMe> >& traceme)
            { return name == std::get<0>(traceme); });
    TORCH_CHECK(tracemes_iter != tracemes.end(), "Push and pop do not match");

    tracemes.erase(tracemes_iter);
  }

  thread_event_lists consolidate() {
    std::lock_guard<std::mutex> g(state_mutex_);

    thread_event_lists result{};
    if (collection_data_.empty()) {
      return result;
    }

    TORCH_CHECK(cnlight::trace::ExportEvents(&collection_data_, &memory_events_, &elase_events_),
                "Transform failed");

    std::unordered_map<int64_t/*thread_id*/, RangeProfilerMluEvent> event_lists_map;
    for (auto& elase_event : elase_events_) {
      int64_t thread_id = elase_event.end_thread_id_;
      int64_t corr_id = elase_event.correlation_id_;

      RangeProfilerMluEvent& thread_event = event_lists_map[thread_id];
      ProfilerMluContext& mlu_context = context_map_[corr_id];

      thread_event.record(corr_id, std::move(elase_event), &mlu_context);
    }

    for (auto& kv : event_lists_map) {
      auto& list = kv.second;
      result.emplace_back(list.consolidate());
    }
    return result;
  }

  void reportMemoryUsage(void* /* unused */,
      int64_t alloc_size, c10::Device device) override {
    if (config_.profile_memory && !tracemes.empty() &&
        config_.state == torch::autograd::profiler::ProfilerState::MLU) {
      ProfilerMemoryEvent& memory_event =
                context_map_[std::get<1>(tracemes.front())].getMemoryEvent();
      memory_event.updateMemoryStats(alloc_size, device);
    }
  }

  bool memoryProfilingEnabled() const override {
    return config_.profile_memory;
  }

  ~MLUThreadLocalState() override {
    delete tracer_;
  }

 private:
  std::mutex state_mutex_;
  std::unordered_map<int64_t, ProfilerMluContext> context_map_;

  torch::autograd::profiler::ProfilerConfig config_ =
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::MLU, false, false);
  at::CallbackHandle handle_ = 0;
  const char* tracer_output_;
  std::string collection_data_;
  std::vector<cnlight::trace::MemoryEvent> memory_events_;
  std::vector<cnlight::trace::ElapseEvent> elase_events_;
  std::list<std::tuple<at::StringView, int64_t,
          std::shared_ptr<cnlight::trace::AnnotatedTraceMe> > > tracemes;

  cnlight::trace::TracerInterface* tracer_{nullptr};
};


MLUThreadLocalState* getProfilerTLSState() {
  const auto& state = c10::ThreadLocalDebugInfo::get(
      c10::DebugInfoKind::PROFILER_STATE);
  return dynamic_cast<MLUThreadLocalState*>(state.get());
}


void pushProfilingCallbacks() {
  auto state_ptr = getProfilerTLSState();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) {
        auto state_ptr = getProfilerTLSState();

        if (!state_ptr || state_ptr->config().state !=
              torch::autograd::profiler::ProfilerState::MLU) {
          return;
        }

        auto corr_id = next_correlation_id();

        if (state_ptr->config().report_input_shapes) {
          std::vector<std::vector<int64_t>> inputSizes;
          inputSizes.reserve(fn.inputs().size());
          for (const c10::IValue& input : fn.inputs()) {
            if (!input.isTensor()) {
              inputSizes.emplace_back();
              continue;
            }
            const at::Tensor& tensor = input.toTensor();
            if (tensor.defined()) {
              inputSizes.push_back(input.toTensor().sizes().vec());
            } else {
              inputSizes.emplace_back();
            }
          }
          state_ptr->pushContext(
            fn.name(), std::to_string(corr_id), std::move(inputSizes), fn.handle(),
            at::RecordFunction::getDefaultNodeId());
        } else {
          state_ptr->pushContext(
            fn.name(), std::to_string(corr_id), {}, fn.handle(),
            at::RecordFunction::getDefaultNodeId());
        }
      },
      [](const at::RecordFunction& fn) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr || state_ptr->config().state !=
            torch::autograd::profiler::ProfilerState::MLU) {
          return;
        }
        state_ptr->popContext(fn.name());
      }
  )
  .needsInputs(state_ptr->config().report_input_shapes)
  .needsIds(true));
  state_ptr->setCallbackHandle(handle);
}

}  // namespace


void reportMluMemoryUsageToProfiler(void* ptr, int64_t alloc_size, c10::Device device) {
  const auto& state = c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE);
  auto* reporter_ptr = static_cast<c10::MemoryReportingInfoBase*>(state.get());
  if (reporter_ptr) {
    reporter_ptr->reportMemoryUsage(ptr, alloc_size, device);
  }
}

void enableMluProfiler(
    const torch::autograd::profiler::ProfilerConfig& config) {
  TORCH_CHECK(config.state == torch::autograd::profiler::ProfilerState::MLU,
    "Can't use Mlu profiler - PyTorch was compiled without Mlu");
  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(!state_ptr, "Profiler is already enabled on this thread");
  auto state = std::make_shared<MLUThreadLocalState>(config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  pushProfilingCallbacks();
  g_.emplace_back(std::make_shared<at::RecordFunctionGuard>());

  const char* host_tracer_level =
      (getenv("HOST_TRACER_LEVEL") != NULL) ? getenv("HOST_TRACER_LEVEL") : "1";
  const char* device_tracer_level =
      (getenv("DEVICE_TRACER_LEVEL") != NULL) ? getenv("DEVICE_TRACER_LEVEL") : "1";
  const char* tracer_output =
      (getenv("MLU_PROFILER_LOGDIR") != NULL) ? getenv("MLU_PROFILER_LOGDIR") : "logdir";

  cnlight::trace::TracerOptions options {
    getHostTracerLevel(host_tracer_level),
    getDeviceTracerLevel(device_tracer_level)
  };

  auto tracer = cnlight::trace::CreateTracer(options, tracer_output);
  TORCH_CHECK(tracer->SetAnnotationDelimiter(":::"),
    "Annotation delimiter is not set successfully");
  state->setTracer(tracer, tracer_output);
  state->getTracer()->Start();
}

thread_event_lists disableMluProfiler() {
  auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
  auto state_ptr = static_cast<MLUThreadLocalState*>(state.get());
  TORCH_CHECK(state_ptr && state_ptr->config().state ==
        torch::autograd::profiler::ProfilerState::MLU,
    "Can't disable profiler when it's not running");

  state_ptr->getTracer()->Stop(state_ptr->getCollectionPtr());
  cnlight::trace::ExportToTensorboard(*state_ptr->getCollectionPtr(),
                      state_ptr->getTracerOutPath());
  g_.pop_back();
  at::removeCallback(state_ptr->callbackHandle());

  return state_ptr->consolidate();
}

}  // namespace profiler
}  // namespace torch_mlu
