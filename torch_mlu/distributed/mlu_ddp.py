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

from datetime import timedelta
import warnings

import torch
import torch.distributed as dist
import torch_mlu.core.mlu_model as ct

# Default process group wide timeout, if applicable.
# This currently only applies to the gloo backend. To make an attempt at
# backwards compatibility with THD, we use an extraordinarily high default
# timeout, given that THD did not have timeouts.
_default_pg_timeout = timedelta(minutes=30)


def is_initialized():
    """
    Keep for backward compatibility

    Checking if the default process group has been initialized

    """

    warnings.warn("torch_mlu.distributed.is_initialized is deprecated, please use "
                  "torch.distributed.is_initialized instead")

    return dist.is_initialized()

def get_mlu_default_group():
    """
    Keep for backward compatibility

    Getting the default process group created by init_process_group

    """

    warnings.warn("torch_mlu.distributed.get_mlu_default_group is deprecated")

    return torch.distributed.distributed_c10d._get_default_group()


def get_world_size():
    """
    Keep for backward compatibility

    Returns the number of processes in the default process group

    Returns:
        The world size of the process group

    """

    warnings.warn("torch_mlu.distributed.get_world_size is deprecated, please use "
                  "torch.distributed.get_world_size instead")

    return dist.get_world_size()


def get_rank():
    """
    Keep for backward compatibility

    Returns the rank of default process group

    """

    warnings.warn("torch_mlu.distributed.get_rank is deprecated, please use "
                  "torch.distributed.get_rank instead")

    return dist.get_rank()


def broadcast(tensor, src):
    """
    Keep for backward compatibility

    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.

    Returns:
        None

    """

    warnings.warn("torch_mlu.distributed.broadcast is deprecated, please use "
                  "torch.distributed.broadcast instead")

    dist.broadcast(tensor, src)

def init_process_group(init_method=None,
                       timeout=_default_pg_timeout,
                       world_size=-1,
                       rank=-1,
                       store=None):
    """
    Keep for backward compatibility

    Initializes the default distributed process group, and this will also
    initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.
        If neither is specified, ``init_method`` is assumed to be "env://".


    Arguments:
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process.
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.

    """

    warnings.warn("torch_mlu.distributed.init_process_group is deprecated, please use "
                  "torch.distributed.init_process_group instead")

    dist.init_process_group(backend="cncl", init_method=init_method, timeout=timeout,
        world_size=world_size, rank=rank, store=store)

def destroy_process_group():
    """
    Keep for backward compatibility

    Destroy default process group, and deinitialize the distributed package

    """

    warnings.warn("torch_mlu.distributed.destroy_process_group is deprecated, please use "
                  "torch.distributed.destroy_process_group instead")

    if is_initialized():
        dist.destroy_process_group()

class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    r"""
    Keep for backward compatibility

    """
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False):

        warnings.warn("torch_mlu.distributed.DistributedDataParallel is deprecated, please use "
                      "torch.nn.parallel.DistributedDataParallel instead")

        device_ids = [ct.current_device()]

        torch.nn.parallel.DistributedDataParallel.__init__(self, module, device_ids=device_ids,
            output_device=output_device, dim=dim, broadcast_buffers=broadcast_buffers,
            process_group=process_group, bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters, check_reduction=check_reduction)
