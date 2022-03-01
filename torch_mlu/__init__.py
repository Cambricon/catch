import os

import torch
import torch_mlu._MLUC as _MLUC

_MLUC._catch_register_function()

def get_version():
    return _MLUC._get_version()

__version__=get_version()

from torch_mlu.core import mlu_model
torch.mlu = mlu_model
