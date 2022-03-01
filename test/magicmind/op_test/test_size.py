from __future__ import print_function
from itertools import product
import torch
import torch.nn as nn
import torch_mlu
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import sys
import os
import copy
import random

import time
import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestSizeModel(nn.Module):
    def __init__(self, dim):
        super(TestSizeModel, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        z = x.size(self.dim)
        # TODO(wangyan): test when mm fixed
        return z + y

class TestSizeOp(TestCase):
    @testinfo()
    def test_size(self):
        dim_l = [0, 3]
        for dim in dim_l:
            for element_type in [torch.half, torch.float, torch.int, torch.short, \
                                 torch.long, torch.uint8, torch.int8, torch.bool]:
                model = TestSizeModel(dim)
                input_x = torch.rand((3,6,8,12)).to(dtype=element_type)
                input_y = torch.randn((3,6,8,12))
                traced_model = torch.jit.trace(model, (input_x, input_y), check_trace=False)
                out_cpu = model(input_x, input_y)

                input_x = input_x.to(dtype=element_type)
                input_x_mlu = input_x.to('mlu')
                input_y_mlu = input_y.to('mlu')
                 
                out_mlu = traced_model(input_x_mlu, input_y_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

if __name__ == '__main__':
    unittest.main()