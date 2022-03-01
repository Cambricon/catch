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

class Queue(torch_mlu._MLUC._Queue):
    r"""Wrapper around a MLU Queue
    An MLU Queue is a series of operations published on the host side to perform
    on an MLU device.

    Arguments:
        device_index(int, optional): Construct a Queue for the curent or specified
            device. If device_index has a value of -1, the current device's
            queue will be used. The default value is -1.
    """
    def __init__(self, device_index = -1):
        super(Queue, self).__init__(device_index)
        self.current_queue = None
        self.device_index = device_index

    def __enter__(self):
        current_device = torch_mlu._MLUC._get_device()
        self.current_queue = torch_mlu._MLUC._getCurrentQueue(current_device)
        user_queue = torch_mlu._MLUC._getQueueFromPool(self.device_index)
        torch_mlu._MLUC._setCurrentQueue(user_queue)
        if (current_device != self.device_index):
            torch_mlu._MLUC._set_device(self.device_index)

    def __exit__(self, type, value, traceback):
        torch_mlu._MLUC._setCurrentQueue(self.current_queue)
        if (self.current_queue.device_index != self.device_index):
            torch_mlu._MLUC._set_device(self.current_queue.device_index);

