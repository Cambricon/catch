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

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "aten/util/exceptions.h"
#include "aten/device/device.h"
#include "aten/device/queue.h"
#include "aten/device/guard_impl.h"
#include "aten/util/common.h"
#include "cnrt.h"  // NOLINT

using torch_mlu::Queue;

namespace torch_mlu { namespace mlu {

struct MLUQueueGuard {
    MLUQueueGuard() = delete;

    explicit MLUQueueGuard(Queue queue)
        : current_queue_(queue), original_queue_(getCurrentQueue()) {
            reset_queue(queue);
        }

    ~MLUQueueGuard() {
        setCurrentQueue(original_queue_);
    }

    MLUQueueGuard(const MLUQueueGuard&) = delete;
    MLUQueueGuard& operator=(const MLUQueueGuard&) = delete;

    MLUQueueGuard(const MLUQueueGuard&& other) = delete;
    MLUQueueGuard& operator=(const MLUQueueGuard&& other) = delete;

    void reset_queue(Queue queue) {
        current_queue_ = queue;
        setCurrentQueue(queue);
    }

    Queue original_queue() const { return original_queue_;  }

    Queue current_queue() const { return current_queue_;  }

    private:
    Queue current_queue_;
    Queue original_queue_;
};


}  // namespace mlu
}  // namespace torch_mlu
