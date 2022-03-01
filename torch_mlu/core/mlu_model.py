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
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

import warnings
import pickle
import torch
import torch_mlu

from torch_mlu.core.device.queue import Queue
from torch_mlu.core.device.device import Device
from torch_mlu.core.device.queue import Queue as Stream
from torch_mlu.core.device.device import Device as _device

### torch.mlu

_device_t = Union[_device, str, int, None]


def init():
    r"""Initialize PyTorch's MLU state.
    """
    return


def is_initialized():
    r"""Returns whether PyTorch's MLU state has been initialized."""
    return True


def is_available():
    r"""Returns a bool indicating if CUDA is currently available."""
    return torch_mlu._MLUC._device_count() > 0


### Device management


def set_device(device):
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``MLU_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = torch.cuda._utils._get_device_index(device, optional=True)
    if device >= 0:
        torch_mlu._MLUC._set_device(device)


def current_device():
    r"""Returns the current device id.

    Args:
        None.
    """
    return torch_mlu._MLUC._current_device()


def device_count():
    r"""Returns the device count.

    Args:
        None.
    """
    return torch_mlu._MLUC._device_count()


class device_of(_device):
    r"""Context-manager that changes the current device to that of given object.

    You can use tensors as arguments. If a given object is
    not allocated on a MLU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """
    def __init__(self, obj):
        idx = obj.get_device() if obj.device.type == 'mlu' else -1
        super(device_of, self).__init__(idx)


def get_device_properties(device: _device_t):
    r"""Gets the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _MLUDeviceProperties: the properties of the device
    """
    device = torch.cuda._utils._get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return torch_mlu._MLUC._get_device_properties(device)


def get_device_capability(
        device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Gets the cuda capability of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.mlu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device. eg. MLU370
    """
    return get_device_properties(device).name


def synchronize(device: Optional[_device_t] = None):
    r"""Waits for all kernels in all streams on a MLU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device_index = torch.cuda._utils._get_device_index(device, optional=True)
    with _device(device_index):
        torch_mlu._MLUC._synchronize()


device = _device

### Stream and Event


def current_stream(device=None):
    device_index = torch.cuda._utils._get_device_index(device, optional=True)
    return torch_mlu._MLUC._getCurrentQueue(device_index)


def default_stream(device=None):
    device_index = torch.cuda._utils._get_device_index(device, optional=True)
    return torch_mlu._MLUC._getDefaultQueue(device_index)


#### Memory management
def empty_cache():
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other MLU application and visible in
    `cnmon info`.
    """
    torch_mlu._MLUC._empty_cached_memory()


def memory_allocated(device: Union[torch.device, str, None, int] = None):
    r"""Returns the current MLU memory occupied by tensors in bytes for a given
    device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device_index = torch.cuda._utils._get_device_index(device, optional=True)
    return torch_mlu._MLUC._memory_allocated(device_index)


def max_memory_allocated(device: Union[torch.device, str, None, int] = None):
    r"""Returns the maximum MLU memory occupied by tensors in bytes for a given
    device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device_index = torch.cuda._utils._get_device_index(device, optional=True)
    return torch_mlu._MLUC._max_memory_allocated(device_index)


def memory_reserved(device: Union[torch.device, str, None, int] = None):
    r"""Returns the current MLU memory managed by the caching allocator in bytes
    for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    """
    device_index = torch.cuda._utils._get_device_index(device, optional=True)
    return torch_mlu._MLUC._memory_cached(device_index)


def max_memory_reserved(device: Union[torch.device, str, None, int] = None):
    r"""Returns the maximum MLU memory managed by the caching allocator in bytes
    for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    """
    device_index = torch.cuda._utils._get_device_index(device, optional=True)
    return torch_mlu._MLUC._max_memory_cached(device_index)


def memory_cached(device: Union[torch.device, str, None, int] = None):
    r"""Deprecated; see :func:`~torch.mlu.memory_reserved`."""
    return memory_reserved(device)


def max_memory_cached(device: Union[torch.device, str, None, int] = None):
    r"""Deprecated; see :func:`~torch.mlu.max_memory_reserved`."""
    return max_memory_reserved(device)


### Torch Module

T = TypeVar('T', bound='Module')


def module_mlu(self: T, device: Optional[Union[int, _device]] = None) -> T:
    if device is None:
        device = current_device()
    device = torch.device(device)
    device = torch.device('mlu', device.index)
    return self._apply(lambda t: t.to(device))


torch.nn.Module.mlu = module_mlu

### Torch Tensor

T = TypeVar('T', bound='Tensor')


def tensor_mlu(self: T,
               device: Optional[Union[int, _device]] = None,
               non_blocking=False,
               memory_format=torch.preserve_format) -> T:
    if device is None:
        device = current_device()
    device = torch.device(device)
    if device.type == 'cuda':
        device = torch.device('mlu', device.index)
    if device.type != 'mlu':
        raise RuntimeError("Invalid device, must be mlu device")
    return self.to(device=device,
                   non_blocking=non_blocking,
                   memory_format=memory_format)


torch.Tensor.mlu = tensor_mlu

### Torch Tensor Dtype

tensor_dtype_dict = {
    "DoubleTensor": torch.float64,
    "FloatTensor": torch.float32,
    "HalfTensor": torch.float16,
    "ByteTensor": torch.uint8,
    "CharTensor": torch.int8,
    "ShortTensor": torch.int16,
    "IntTensor": torch.int32,
    "LongTensor": torch.int64,
    "BoolTensor": torch.bool,
}


def dtype_tensor_wrap(func):
    def wrap(*args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = torch.device('mlu')
        else:
            kwargs['device'] = torch.device(kwargs['device'])
        if kwargs['device'].type != 'mlu':
            raise RuntimeError("legacy constructor expects device type: mlu")
        if len(args) == 0:
            torch.tensor([], device='mlu')
        if isinstance(args[0], list):
            kwargs['dtype'] = tensor_dtype_dict[func.__name__]
            return torch.tensor(*args, **kwargs)
        kwargs['dtype'] = tensor_dtype_dict[func.__name__]
        return torch.empty(args, **kwargs)

    return wrap


FloatTensor = dtype_tensor_wrap(torch.cuda.FloatTensor)
HalfTensor = dtype_tensor_wrap(torch.cuda.HalfTensor)
ByteTensor = dtype_tensor_wrap(torch.cuda.ByteTensor)
CharTensor = dtype_tensor_wrap(torch.cuda.CharTensor)
ShortTensor = dtype_tensor_wrap(torch.cuda.ShortTensor)
IntTensor = dtype_tensor_wrap(torch.cuda.IntTensor)
BoolTensor = dtype_tensor_wrap(torch.cuda.BoolTensor)
DoubleTensor = dtype_tensor_wrap(torch.cuda.DoubleTensor)
LongTensor = dtype_tensor_wrap(torch.cuda.LongTensor)

### Serialization

torch_load = torch.load


def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    if map_location is None:
        return torch_load(f,
                          map_location=map_location,
                          pickle_module=pickle_module,
                          **pickle_load_args)
    if map_location == 'cpu' or map_location == torch.device('cpu'):
        return torch_load(f,
                          map_location=map_location,
                          pickle_module=pickle_module,
                          **pickle_load_args)
    if not isinstance(map_location, str) and not isinstance(
            map_location, torch.device):
        return torch_load(f,
                          map_location=map_location,
                          pickle_module=pickle_module,
                          **pickle_load_args)

    ckpt = torch_load(f,
                      map_location='cpu',
                      pickle_module=pickle_module,
                      **pickle_load_args)
    dst_device = torch.device(map_location)
    dst_device = torch.device('mlu', dst_device.index)
    if isinstance(ckpt, dict):
        for key in ckpt.keys():
            ckpt[key] = ckpt[key].to(dst_device)
        return ckpt
    return ckpt.to(dst_device)


torch.load = load

### Random sampling


def manual_seed(seed):
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)
    torch_mlu._MLUC._manual_seed(seed)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers for the current MLU.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-MLU model, this function is insufficient
        to get determinism.  To seed all MLUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)
    torch_mlu._MLUC._manual_seed_all(seed)


### Torch MLU


def is_mlu_tensor(tensor):
    r"""Judge the device attribute of input tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
    """
    return tensor.device.type == 'mlu'


def mlu_device():
    r"""Returns a MLU device instance.

    Args:
        None.
    """
    return torch.device('mlu')


def enable_floating_point_calculation(flag):
    r"""
        Change the running mode of floating-point supported device.
    Args:
        flag (bool): flag=True to run fp32 mode,
                     flag=False to run online-quantized mode.
    """
    torch_mlu._MLUC._enable_floating_point_calculation(flag)


def is_using_floating_device():
    r"""Judge floating-point supported device.
    Args:
        None.
    """
    return torch_mlu._MLUC._is_using_floating_device()


def get_running_mode():
    return torch_mlu._MLUC._get_running_mode()


def get_device():
    r"""Returns the device id set by set_device().

    Args:
        None.
    """
    return torch_mlu._MLUC._get_device()

r"""
    Queue Management.
    The interface allows you to get the corresponding queue instance and
    corresponding operations.

    Arguments:
        device_index(int, optional): Returns a Queue for the curent ot specified
            device. If device_index has a value of -1, the current device's
            queue will be used. The default value is -1.
"""


def current_queue(device_index=-1):
    r"""
        Returns the current queue for a given device ID.
    Args:
        device_index(int, optional): returns a queue for the current or specified
            device. If device_index is -1, the current device's queue will be returned.
            The default value is -1.
    """
    return torch_mlu._MLUC._getCurrentQueue(device_index)


def default_queue(device_index=-1):
    r"""
        Returns the default queue for a given device ID.
    Args:
        device_index(int, optional): returns a default queue for the current or specified
            device. If device_index is -1, the current device's default queue will be returned.
            The default value is -1.
    """
    return torch_mlu._MLUC._getDefaultQueue(device_index)


def empty_cached_memory():
    r"""
        cnrtFree all cached memory
    """
    torch_mlu._MLUC._empty_cached_memory()


def to(optimizer, device):
    for state_perparam in optimizer.state.values():
        for k, v in state_perparam.items():
            if isinstance(v, torch.Tensor):
                state_perparam[k] = v.to(device)
    return optimizer


def set_memory_strategy(native_memory_strategy:bool):
    torch_mlu._MLUC._set_memory_strategy(native_memory_strategy)

def memory_debug(tensor=None):
    r"""
        start memory debugging
    """
    if tensor is None:
        return torch_mlu._MLUC._memory_debug()
    else:
        return torch_mlu._MLUC._memory_debug(tensor)


def pin_memory(tensor):
    r"""
       Returns the pinned memory copy of tensor
    """
    return torch_mlu._MLUC._pin_memory(tensor)


def is_pinned(tensor):
    r"""
       Returns the tensor is pinned or not
    """
    return torch_mlu._MLUC._is_pinned(tensor)


def _jit_override_can_fuse_on_mlu(param: 'bool' = True):
    r"""Enable jit IR fuse or not.

    Args:
        param (bool): True/False
    """
    torch_mlu._MLUC._jit_override_can_fuse_on_mlu(param)
