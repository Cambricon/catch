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

#include "interface/python_variable_methods.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>

#include "aten/core/caching_allocator.h"
#include "aten/core/generator_impl.h"
#include "aten/device/device.h"
#include "aten/util/python_interface.h"
#include "aten/util/python_notifier.h"
#include "aten/util/memory_allocator.h"
#include "aten/util/version.h"
#include "aten/util/cndumper.h"
#include "init/init.h"
#include "jit/interface.h"

#if USE_PROFILE
#include "aten/profile/profile_mlu.h"
#endif

namespace torch_mlu {
namespace {

void PythonVariableMethods(py::module& m) {
  // MLU Model Management.
  m.def("_catch_register_function", []() { CatchRegisterFunc(); });
  m.def("_get_running_mode",
        []() -> bool { return PythonInterface::getRunningModeBool(); });
  m.def("_is_using_floating_device", []() -> bool {
          return torch_mlu::Global::instance().isUsingFloatingDevice(); });
  m.def("_set_memory_strategy",
        [](bool native_memory_strategy) { torch_mlu::set_memory_strategy(native_memory_strategy); });

  // Device Management.
  m.def("_set_device", [](int param) { PythonInterface::setDevice(param); });
  m.def("_get_device", []() { return PythonInterface::getDevice(); });
  m.def("_device_count", []() { return PythonInterface::deviceCount(); });
  m.def("_current_device",
        []() -> int { return static_cast<int>(torch_mlu::current_device()); });
  m.def("_synchronize", []() {return torch_mlu::synchronize(); });

  // device properties
  pybind11::class_<torch_mlu::DeviceProp>(m, "_MLUDeviceProperties")
      .def_readonly("name", &DeviceProp::name)
      .def_readonly("major", &DeviceProp::major)
      .def_readonly("minor", &DeviceProp::minor)
      .def_readonly("total_memory", &DeviceProp::total_memory)
      .def("__repr__", [](const torch_mlu::DeviceProp & prop) {
          std::ostringstream stream;
          stream << "_MLUDeviceProperties(name='" << prop.name << "', major=" << prop.major
              << ", minor=" << prop.minor
              << ", total_memory=" << prop.total_memory / (1024 * 1024)
              << "MB)";
          return stream.str();
      });

  m.def("_get_device_properties", [](int device) -> torch_mlu::DeviceProp* {
          return torch_mlu::getDeviceProperties(device);
          }, pybind11::return_value_policy::reference);

  // Enable jit mlu fused kernel
  m.def("_jit_override_can_fuse_on_mlu", &torch_mlu::jit::overrideCanFuseOnMLU);
  m.def("_jit_can_fuse_on_mlu", &torch_mlu::jit::canFuseOnMLU);

  // Queue Management.
  pybind11::class_<torch_mlu::Queue>(m, "_Queue")
      .def(pybind11::init<int>())
      .def("query", &torch_mlu::Queue::query)
      .def("synchronize",
          [](torch_mlu::Queue& _queue) {
            pybind11::gil_scoped_release no_gil;
            _queue.synchronize();
          })
      .def("__eq__", &torch_mlu::Queue::operator==)
      .def_readwrite("device", &torch_mlu::Queue::device_index_)
      .def_readwrite("device_index", &torch_mlu::Queue::device_index_);
  m.def("_getQueueFromPool", [](int param) -> Queue {return torch_mlu::getQueueFromPool(param); });
  m.def("_getDefaultQueue", [](int param) -> Queue {return torch_mlu::getDefaultQueue(param); });
  m.def("_getCurrentQueue", [](int param) -> Queue {return torch_mlu::getCurrentQueue(param); });
  m.def("_setCurrentQueue", [](Queue param) { setCurrentQueue(param); });

  // Memory Management.
  m.def("_memory_allocated", [](int param) -> uint64_t {return
        torch_mlu::currentMemoryAllocated(param); });
  m.def("_memory_cached", [](int param) -> uint64_t {return
        torch_mlu::currentMemoryCached(param); });
  m.def("_max_memory_allocated", [](int param) -> uint64_t {return
        torch_mlu::maxMemoryAllocated(param); });
  m.def("_max_memory_cached", [](int param) -> uint64_t {return
        torch_mlu::maxMemoryCached(param); });
  m.def("_empty_cached_memory", [](){ torch_mlu::emptyCachedMem(); });
  m.def("_current_device", []() -> int { return static_cast<int>(torch_mlu::current_device()); });
  m.def("_memory_debug", [](at::Tensor data){
      torch_mlu::memoryDebug(&data.storage().data_ptr()); });
  m.def("_memory_debug", [](){
      torch_mlu::memoryDebug(); });
  m.def("_pin_memory", [](at::Tensor data){
      pybind11::gil_scoped_release no_gil;
      return torch_mlu::pinMemory(data);
  });
  m.def("_is_pinned", [](at::Tensor data) -> bool { return
      torch_mlu::isPinned<void>(data.data_ptr()); });
  // Notifier Management
  pybind11::class_<torch_mlu::PyNotifier>(m, "_Notifier")
      .def(pybind11::init())
      .def("place", (void (torch_mlu::PyNotifier::*)(torch_mlu::Queue&)) &
                        torch_mlu::PyNotifier::place,
           "place notifier on queue")
      .def("elapsed_time", &torch_mlu::PyNotifier::elapsed_time)
      .def("hardware_time", &torch_mlu::PyNotifier::hardware_time)
      .def("synchronize",
          [](torch_mlu::PyNotifier& _notifier) {
            pybind11::gil_scoped_release no_gil;
           _notifier.synchronize();
          })
      .def("query", &torch_mlu::PyNotifier::query)
      .def("wait",
          [](torch_mlu::PyNotifier& _notifier, torch_mlu::Queue& queue) {
            pybind11::gil_scoped_release no_gil;
            _notifier.wait(queue);
          });

  // Dumptools API
  m.def("_dump_start", [&](const char* dump_dir, bool enable, bool use_cpu, int level) {
        torch_mlu::global_dumptool.startDump(dump_dir, enable, use_cpu, level); });
  m.def("_dump_finish", [&]() {torch_mlu::global_dumptool.endDump(); });
  m.def("_dump_cnnl_gencase", [&](int mode) {torch_mlu::_dump_cnnl_gencase(mode); });
  m.def("_get_version", []() { return torch_mlu::getVersion(); });
  m.def("_enable_floating_point_calculation", [](bool flag) {
        torch_mlu::Global::instance().setFP32RunningMode(flag);});

  // Generate pseudorandom numbers API
  m.def("_manual_seed", [](uint64_t seed) {
        torch_mlu::manual_seed(seed);});

  m.def("_manual_seed_all", [](uint64_t seed) {
        torch_mlu::manual_seed_all(seed);});

  // profiler API
  #if USE_PROFILE
  pybind11::class_<torch_mlu::profiler::DeviceEventVisitor>(m, "DeviceEvent")
      .def("name", &torch_mlu::profiler::DeviceEventVisitor::name)
      .def("start_ns", &torch_mlu::profiler::DeviceEventVisitor::start_ns)
      .def("duration_ns", &torch_mlu::profiler::DeviceEventVisitor::duration_ns)
      .def("device_index", &torch_mlu::profiler::DeviceEventVisitor::device_index);

  pybind11::class_<torch_mlu::profiler::ElapseEventVisitor>(m, "ElapseEvent")
      .def("id", &torch_mlu::profiler::ElapseEventVisitor::id)
      .def("type", &torch_mlu::profiler::ElapseEventVisitor::type)
      .def("start_ns", &torch_mlu::profiler::ElapseEventVisitor::start_ns)
      .def("end_ns", &torch_mlu::profiler::ElapseEventVisitor::end_ns)
      .def("start_thread_id", &torch_mlu::profiler::ElapseEventVisitor::start_thread_id)
      .def("end_thread_id", &torch_mlu::profiler::ElapseEventVisitor::end_thread_id)
      .def("parent", &torch_mlu::profiler::ElapseEventVisitor::parent)
      .def("children", &torch_mlu::profiler::ElapseEventVisitor::children)
      .def("device_events", &torch_mlu::profiler::ElapseEventVisitor::device_events);

  pybind11::class_<torch_mlu::profiler::ProfilerMemoryEvent>(m, "MemoryEvent")
      .def("getCpuMemoryUsage", &torch_mlu::profiler::ProfilerMemoryEvent::getCpuMemoryUsage)
      .def("getMluMemoryUsage", &torch_mlu::profiler::ProfilerMemoryEvent::getMluMemoryUsage);

  pybind11::class_<torch_mlu::profiler::ProfilerMluEvent>(m, "ProfilerMluEvent")
      .def("elapse_event", &torch_mlu::profiler::ProfilerMluEvent::elapse_event)
      .def("cpu_memory_usage", &torch_mlu::profiler::ProfilerMluEvent::cpu_memory_usage)
      .def("mlu_memory_usage", &torch_mlu::profiler::ProfilerMluEvent::mlu_memory_usage)
      .def("shapes", &torch_mlu::profiler::ProfilerMluEvent::shapes)
      .def("handle", &torch_mlu::profiler::ProfilerMluEvent::handle)
      .def("node_id", &torch_mlu::profiler::ProfilerMluEvent::node_id);

  m.def("_enable_mlu_profiler", &torch_mlu::profiler::enableMluProfiler);
  m.def("_disable_mlu_profiler", &torch_mlu::profiler::disableMluProfiler);
  #endif
}
}  // namespace

}  // namespace torch_mlu

// Init methods of py::module
void initMLUModule(PyObject* m) {
  auto t = py::handle(m).cast<py::module>();
  torch_mlu::PythonVariableMethods(t);
}
