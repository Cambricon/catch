from __future__ import print_function

import torch
import torch_mlu

# MLU Profiler is a tool that allows the collecton of the performance metrics during the training and inference.
# Profiler’s context manager API can be used to better understand what model operators are the most expensive,
# examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

def _enable_mlu_profiler(config):
    r"""Enable the mlu profiler.

    Args: None
    """
    assert isinstance(config, torch.autograd.ProfilerConfig), "config must be ProfilerState type"
    torch_mlu._MLUC._enable_mlu_profiler(config)

def _disable_mlu_profiler():
    r"""Disable the mlu profiler.

    Args: None
    """
    return torch_mlu._MLUC._disable_mlu_profiler()
