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

class TestToModel(nn.Module):
    def __init__(self, element_type):
        super(TestToModel, self).__init__()
        self.dtype=element_type

    def forward(self, x):
        z = x.to(dtype=self.dtype)
        return z


class TestToOp(TestCase):
    @testinfo()
    def test_to(self):
        from_types = [torch.float, torch.half, torch.int, torch.short, \
                      torch.uint8, torch.int8]
        to_types = [torch.float, torch.half, torch.int, torch.short, \
                    torch.uint8, torch.int8, torch.bool]

        for from_dtype in from_types:
            for to_dtype in to_types:
                if from_dtype is torch.int and to_dtype is torch.uint8:
                    continue
                if from_dtype is torch.short and \
                    (to_dtype not in [torch.float, torch.half, torch.short]):
                    continue
                if from_dtype is torch.uint8 and \
                    (to_dtype not in [torch.float, torch.half]):
                    continue
                if from_dtype is torch.int8 and \
                    (to_dtype not in [torch.float, torch.half, torch.int]):
                    continue

                model = TestToModel(to_dtype)
                input_x = torch.rand((3,6,8,12)).to(dtype=from_dtype)
                traced_model = torch.jit.trace(model, (input_x), check_trace=False)
                input_x_mlu = input_x.to('mlu')              

                out_cpu = model(input_x)
                out_mlu = traced_model(input_x_mlu)
                if to_dtype is torch.half:
                    self.assertTensorsEqual(out_cpu.float(),out_mlu.cpu().float(), 0.0)
                else:
                    self.assertTensorsEqual(out_cpu,out_mlu.cpu(), 0)
        
        # test for bool input
        input_x = torch.rand((3,6,8,12)).to(dtype=torch.bool)
        for to_dtype in [torch.float, torch.half, torch.int]:
                model = TestToModel(to_dtype)
                traced_model = torch.jit.trace(model, (input_x), check_trace=False)
                input_x_mlu = input_x.to('mlu')              

                out_cpu = model(input_x)
                out_mlu = traced_model(input_x_mlu)
                if to_dtype is torch.half:
                    self.assertTensorsEqual(out_cpu.float(),out_mlu.cpu().float(), 0.0)
                else:
                    self.assertTensorsEqual(out_cpu,out_mlu.cpu(), 0)


if __name__ == '__main__':
    unittest.main()
