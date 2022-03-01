# coding=utf-8
import warnings
import numpy as np
import torch
import math
import torch.nn as nn
import torch_mlu.core.mlu_model as ct
from abc import ABCMeta, abstractmethod
from functools import partial
from functools import wraps
from .qscheme import mlu_qscheme

def counter(attribute):
    class Counter(object):
        def __init__(self, func):
            self.func = func

        def __get__(self, instance, cls):
            if instance is None:
                return self
            return self.make_bound(instance)

        def make_bound(self, instance):
            instance.ncalls = 0
            @wraps(self.func)
            def wrapper(*args, **kwargs):
                count = getattr(instance, attribute)()
                if count > instance.ncalls:
                    instance.ncalls += 1
                    return self.func(instance, *args, **kwargs)
                else:
                    print("[warning] It seems that evaluation reaches maxium img_num or iteration is in network. Quantization still works.")
            setattr(instance, self.func.__name__, wrapper)
            return wrapper
    return Counter

class _PartialWrapper(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, *args, **keywords):
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__()

def _with_args(cls_or_self, **kwargs):
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r

_PartialWrapper.with_args = _with_args

ABC = ABCMeta(str("ABC"), (object,), {}) # compatible with Python 2 *and* 3:

class Observer(ABC, nn.Module):
    '''
    Observer base Module
    '''
    def __init__(self, dtype):
        super(Observer, self).__init__()
        self.dtype = dtype

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)
class _MLUObserverBase(Observer):
    '''
    common base for mlu observer
    '''
    def __init__(self, dtype='int8', qscheme=mlu_qscheme):
        super(_MLUObserverBase, self).__init__(dtype=dtype)
        self.qscheme = dict(mlu_qscheme)
        self.qscheme.update(qscheme)
        self.ncalls = 0

    def get_count(self):
        return self.qscheme['iteration']

    def _calculate_per_channel_qparams(self, min_vals, max_vals):
        # type: (Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Given min and max value tensors, this function calculates per channel
        quantization parameters
        """
        if min_vals is None or max_vals is None:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        for i in range(len(min_vals)):
            assert (
                min_vals[i] <= max_vals[i]
            ), "min {} should be less than max {}".format(min_vals[i], max_vals[i])

        scales = [1 for _ in range(len(min_vals))]

        for i in range(len(scales)):
            qparam = self._calculate_qparams(
                min_vals[i], max_vals[i]
            )
            scales[i] = float(qparam)

        return scales

    def _calculate_qparams(self, min_val, max_val):
        if max_val is None or min_val is None:
            warnings.warn("Must run observer before calling calculate_qparams.\
                          Returning default scale and zero point")
            return 1.0

        if math.isnan(max_val) or math.isnan(min_val):
            warnings.warn("Your network weight or input have nan value,\
                which can't be quantized.")
            return 1.0

        assert min_val <= max_val, "min {} should be less than max {}".format(min_val, max_val)

        if self.dtype == 'int8':
            qmin, qmax = -128.0, 127.0
        elif self.dtype == 'int16':
            qmin, qmax = -32768.0, 32767.0
        else:
            qmin, qmax = -128.0, 127.0
            warnings.warn("Data type must be int8 or int16.\
                          Using default int8 data type")

        max_val, min_val = float(max_val), float(min_val)
        scale = 1.0
        absmax = max(max_val, -min_val)
        if absmax != 0:
            scale = qmax/ absmax
        return scale

class MLUMinMaxObserver(_MLUObserverBase):
    def __init__(self, dtype='int8', qscheme=mlu_qscheme):
        super(MLUMinMaxObserver, self).__init__(dtype=dtype, qscheme=qscheme)
        self.min_val = None
        self.max_val = None
        self.scale = None

    @counter('get_count')
    def forward(self, *args):
        x_orig = args[0]
        x = x_orig.detach()
        ncalls = self.ncalls
        imin = torch.min(x).item() * self.qscheme['data_scale']
        imax = torch.max(x).item() * self.qscheme['data_scale']
        if self.min_val is None or self.max_val is None:
            self.min_val = imin
            self.max_val = imax
        elif self.qscheme['use_avg']:
            self.min_val = (self.min_val * float(ncalls) + imin) / (float(ncalls) + 1)
            self.max_val = (self.max_val * float(ncalls) + imax) / (float(ncalls) + 1)
        else:
            self.min_val = min(imin, self.min_val)
            self.max_val = max(imax, self.max_val)

    def calculate_qparams(self):
        self.scale = self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

class MLULrnObserver(_MLUObserverBase):
    def __init__(self, dtype='int8', qscheme=mlu_qscheme):
        super(MLULrnObserver, self).__init__(dtype=dtype, qscheme=qscheme)
        self.min_val = None
        self.max_val = None
        self.scale = None

    @counter('get_count')
    def forward(self, *args):
        x_orig = args[0]
        x = x_orig.detach()
        ncalls = self.ncalls
        imin = torch.min(x).item() * self.qscheme['data_scale']
        imax = torch.max(x).item() * self.qscheme['data_scale']
        if self.min_val is None or self.max_val is None:
            self.min_val = imin
            self.max_val = imax
        elif self.qscheme['use_avg']:
            self.min_val = (self.min_val * float(ncalls) + imin) / (float(ncalls) + 1)
            self.max_val = (self.max_val * float(ncalls) + imax) / (float(ncalls) + 1)
        else:
            self.min_val = min(imin, self.min_val)
            self.max_val = max(imax, self.max_val)

    def calculate_qparams(self):
        self.scale = self._calculate_qparams(self.min_val**2, self.max_val**2)

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

class MLUPerChannelMinMaxObserver(_MLUObserverBase):
    r"""Per Channel Observer Module
    The module will record the running average of max and min value for each
    channel of the observed Tensor and calculate_qparams will calculate
    scales and zero_points for each channel
    """

    def __init__(self, dtype='int8', qscheme=mlu_qscheme, ch_axis=0):
        super(MLUPerChannelMinMaxObserver, self).__init__(dtype=dtype, qscheme=qscheme)
        self.min_vals = None
        self.max_vals = None
        self.scales = None
        self.ch_axis = ch_axis

    @counter('get_count')
    def forward(self, x_orig):
        if self.dtype in {'int8', 'int16'}:
            x = x_orig.detach()  # avoid keeping autograd tape
            ncalls = self.ncalls
            y = torch.flatten(x, start_dim = 1)
            imin = torch.min(y, 1)[0] * self.qscheme['data_scale']
            imax = torch.max(y, 1)[0] * self.qscheme['data_scale']
            if self.min_vals is None or self.max_vals is None:
                self.min_vals = imin
                self.max_vals = imax
            else:
                self.min_vals = torch.min(imin, self.min_vals)
                self.max_vals = torch.max(imax, self.max_vals)
            self.min_vals = self.min_vals.tolist()
            self.max_vals = self.max_vals.tolist()
            return x_orig

    def calculate_qparams(self):
        self.scales = self._calculate_per_channel_qparams(self.min_vals, self.max_vals)

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_vals, self.max_vals)

class NoopObserver(Observer):
    def __init__(self, dtype=torch.float16):
        if dtype != torch.float16:
            raise ValueError("Only float16 quantization can be used without calibration process")
        super(NoopObserver, self).__init__(dtype=dtype)

    def forward(self, x):
        return x

    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for NoopObserver")

class MLUNormObserver(_MLUObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values for MLU device.
    """
    def __init__(self, dtype='int8', qscheme=mlu_qscheme):
        super(MLUNormObserver, self).__init__(dtype=dtype, qscheme=qscheme)
        self.boundary_1 = None
        self.boundary_2 = None
        self.scale = None

    @counter('get_count')
    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""

        if self.dtype in {'int8', 'int16'}:
            x = x_orig.detach()  # avoid keeping autograd tape
            max_value = x.max().item()
            broadcast_t = x * (1.0 / max_value)

            conv1_input = broadcast_t.pow(2)
            conv2_input = broadcast_t.pow(2).sum(dim=1).sqrt() + 1e-10
            conv1_max_value = conv1_input.max().item()
            conv1_min_value = conv1_input.min().item()
            conv2_max_value = conv2_input.max().item()
            conv2_min_value = conv2_input.min().item()
            if self.boundary_1 is None or self.boundary_2 is None:
                self.boundary_1 = [conv1_min_value, conv1_max_value]
                self.boundary_2 = [conv2_min_value, conv2_max_value]
            else:
                self.boundary_1 = [min(self.boundary_1[0], conv1_min_value),
                                   max(self.boundary_1[1], conv1_max_value)]
                self.boundary_2 = [min(self.boundary_2[0], conv2_min_value),
                                   max(self.boundary_2[1], conv2_max_value)]

    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        scale1 = self._calculate_qparams(*self.boundary_1)
        scale2 = self._calculate_qparams(*self.boundary_2)
        self.scale = [scale1, scale2]

    def extra_repr(self):
        return "boundary_1={}, boundary_2={}".format(self.boundary_1, self.boundary_2)


default_mlu_observer = MLUMinMaxObserver.with_args(dtype='int8', qscheme=mlu_qscheme)

def mlu_weight_observer(dtype):
    return MLUMinMaxObserver.with_args(dtype=dtype)

def mlu_activation_observer(dtype, qscheme):
    return MLUMinMaxObserver.with_args(dtype=dtype, qscheme=qscheme)

def mlu_minmax_observer(dtype, qscheme):
    r""" Initialize activation observer
    """
    return MLUMinMaxObserver.with_args(dtype=dtype, qscheme=qscheme)

def mlu_norm_observer(dtype, qscheme):
    return MLUNormObserver.with_args(dtype=dtype, qscheme=qscheme)

def mlu_lrn_observer(dtype, qscheme):
    return MLULrnObserver.with_args(dtype=dtype, qscheme=qscheme)

def mlu_weight_per_channel_observer(dtype, qscheme):
    r""" Initialize weight per channel observer

    Note that for weight per channel observer, qscheme is not needed in calculating scale
    """
    return MLUPerChannelMinMaxObserver.with_args(dtype=dtype, qscheme=qscheme)

mlu_activation_observer_dict = {'minmax': mlu_minmax_observer}

