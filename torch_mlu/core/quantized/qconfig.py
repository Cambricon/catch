from collections import namedtuple
from  .observer import *
import torch.nn as nn

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    # Both activation and weight will be attached with qconfig
    def __new__(cls, activation, weight):
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        return super(QConfig, cls).__new__(cls, activation, weight)

default_mlu_qconfig = QConfig(activation=default_mlu_observer,
                          weight=default_mlu_observer)

def mlu_weight_qconfig(dtype):
    return QConfig(weight = mlu_weight_observer(dtype),
                   activation = NoopObserver)

def mlu_activation_qconfig(dtype, qscheme):
    return QConfig(weight = NoopObserver,
                   activation = mlu_activation_observer(dtype, qscheme))

def custom_mlu_qconfig(dtype, qscheme):
    return QConfig(weight = mlu_weight_observer(dtype),
                   activation = mlu_activation_observer(dtype, qscheme))

def norm_mlu_qconfig(dtype, qscheme):
    return QConfig(weight = mlu_weight_observer(dtype),
                   activation = mlu_norm_observer(dtype, qscheme))

def mlu_per_channel_qconfig(dtype, qscheme):
    return QConfig(weight = mlu_weight_per_channel_observer(dtype, qscheme),
                   activation = mlu_activation_observer(dtype, qscheme))
