'''
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
'''

from __future__ import print_function

import torch
import torch_mlu
import torch_mlu.core.mlu_model as ct

class Notifier(torch_mlu._MLUC._Notifier):
    r"""

    MLU Notifier is a sync mark. It can be used to do the following things:
    1. statistic executime time.
    2. sync the execution process and different queue.

    Arguments: None

    """
    def __init__(self):
        super(Notifier, self).__init__()

    # Place the notifier on the specified queue
    def place(self,queue = None):
        if queue is None:
            queue = ct.current_queue()
        super(Notifier,self).place(queue)

    # Wait the specified queue
    def wait(self,queue = None):
        if queue is None:
            queue = ct.current_queue()
        super(Notifier,self).wait(queue)

    # Query the queue.
    def query(self):
        return super(Notifier,self).query()

    # Returns the end-to-end time between start notifier and end notifier.
    def elapsed_time(self,end_notifier):
        time = super(Notifier,self).elapsed_time(end_notifier)
        return time

    # Returns the hardware time between start notifier and end notifier.
    def hardware_time(self,end_notifier):
        time = super(Notifier,self).hardware_time(end_notifier)
        return time

    # Sync the queue.
    def synchronize(self):
        super(Notifier,self).synchronize()
