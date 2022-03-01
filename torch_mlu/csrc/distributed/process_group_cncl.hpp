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

#include <pybind11/chrono.h>
#include <torch/extension.h>
#include <unordered_map>

#include "distributed/cncl_utils.h"
#include "aten/device/queue_guard.h"
#include "aten/device/notifier.h"
#include "aten/util/tensor_util.h"

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include "c10d/Utils.hpp"


namespace c10d {

// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
constexpr const char* CNCL_BLOCKING_WAIT = "CNCL_BLOCKING_WAIT";

// ProcessGroupCNCL implements CNCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All CNCL functions provided by this class are asynchronous functions. More
// specifically, each CNCL call is scheduled on a separate MLU stream that is
// different from the current MLU stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibilty to make sure that the MLU stream their
// code works on needs to wait for the CNCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkCNCL::wait() or WorkCNCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Note that WorkCNCL::isSuccess() and WorkCNCL::isCompleted() will always
// return true since ProcessGroupCNCL is single threaded. Every single CNCL
// or MLU failure will simply raise std::runtime_error.
//
// Therefore, WorkCNCL::exception() is not supported since isSuccess() always
// returns true.
//
// Also note that WorkCNCL::finishedMLUExecution() is a helper function only
// provided by ProcessGroupCNCL to check if the CNCL operation of WorkCNCL has
// finished execution on the MLU (not just scheduled).
//
// Example on using the CNCL process group
//
//   ProcessGroupCNCL pg(store, rank, size);
//   std::shared_ptr<WorkCNCL> work = pg.allreduce(tensors);
//
//   // At this point, CNCL kernel has already by queued successfully
//   // Now, let current stream wait for the CNCL to finish, originally this function is
//   // async operation as well, but currently MLU is sync.
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class ProcessGroupCNCL : public ProcessGroup {
 public:
  class WorkCNCL : public ProcessGroup::Work {
   public:
    // Constructor takes a list of MLU devices
    WorkCNCL(const std::vector<at::Device>& devices); // NOLINT
    virtual ~WorkCNCL();

    // Checks if request has completed. In this specific case of CNCL, it checks
    // if the CNCL operation has completed on the MLU in its own CNCL queue.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for CNCL work.
    bool wait() override;

    // Let current stream wait on the completing of the CNCL work
    // Throws on exceptions
    void synchronize() override;

   protected:
    // The cached list of MLU devices to operate on
    std::vector<at::Device> devices_;

    // The MLU notifier tracking this work item on MLU device
    torch_mlu::Notifier notifier_;

    // Clone of blockingWait_ from ProcessGroupCNCL.
    bool blockingWait_ = false;

    // Tensors used for barrier op
    std::vector<at::Tensor> barrier_tensors_;

   private:
    // Just checks whether MLU execution has completed, without modifying
    // exception_ptr.
    bool finishedMLUExecutionInternal() const;

    friend class ProcessGroupCNCL;
  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any CNCL communicators. A single CNCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another CNCL
  // communicator. These CNCL communicators are cached and reused if possible.
  ProcessGroupCNCL(
      const std::shared_ptr<Store>& store,
      int rank,
      int size);

  virtual ~ProcessGroupCNCL();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
      AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  // Unsupported Ops
  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

 protected:
  // Helper that broadcasts cncl clique ID to all ranks through the store
  void broadcastCNCLCliqueID(
    cnclCliqueId* cncl_id,
    const bool is_p2p_op,
    const std::string& p2p_key,
    const int p2p_rank);

  // Helper that either looks up the cached CNCL communicators or creates
  // a new set of CNCL communicators as a cache entry
  std::shared_ptr<CNCLComms>& getCNCLComms(
      const std::string& devices_key,
      const std::vector<at::Device>& devices,
      const bool is_p2p_op,
      const int p2p_rank);

  // The store is used to broadcast the CNCL unique ID of rank 0.
  std::shared_ptr<Store> store_;

  // The number of CNCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t cncl_comm_counter_{0};

  // The CNCL communicator that the process group has cached.
  // The key is a list of MLU devices that an operation is operating on
  // The MLU devices are stored in a device sequence and the cache CNCL
  // communicator is associated with this MLU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  std::unordered_map<std::string, std::shared_ptr<CNCLComms>>
      dev_cncl_comm_map_;

  // The MLU queues used by CNCL kernels
  torch_mlu::Queue cncl_queue_;

  // The MLU notifiers used to sync CNCL queues
  torch_mlu::Notifier cncl_notifier_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Create a new ProcessGroupCNCL instance
  static std::shared_ptr<ProcessGroup> createProcessGroupCNCL(
      const std::shared_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void ProcessGroupCNCLConstructor() __attribute__((constructor)) {
      py::object module = py::module::import("torch.distributed");
      py::object register_backend = module.attr("Backend").attr("register_backend");
      // The first parameter is the backend name used by user in invoking
      // torch.distributed.init_process_group().
      register_backend("cncl", py::cpp_function(createProcessGroupCNCL));
  }
};

}  // namespace c10d
