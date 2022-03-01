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

import time
import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestUpsampleModel1(nn.Module):
    def __init__(self, size):
        super(TestUpsampleModel1, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.upsample(x, size=self.size)
        return x

class TestUpsampleModel2(nn.Module):
    def __init__(self, scale_factor):
        super(TestUpsampleModel2, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # TODO(wangyan): check tfu core dump
        z = F.upsample(x + 3, scale_factor=self.scale_factor)
        return z

class TestUpsampleOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_upsample(self):
        shape_l = [(2,3,4,5)]
        size_l = [3]
        scale_factor_l = [2]
        loop_var = [shape_l, size_l, scale_factor_l]
        for shape, size, scale_factor in product(*loop_var):
            model = TestUpsampleModel1(size).eval()
            input_x = torch.randn(shape).float()
            traced_model = torch.jit.trace(model, input_x, check_trace=False)
            input_x_mlu = input_x.to('mlu')
            out_cpu1 = model(input_x)
            out_mlu1 = traced_model(input_x_mlu)
            self.assertTensorsEqual(out_cpu1, out_mlu1.cpu(), 0.003, use_MSE = True)
            # Test for fp16
            traced_model.half()
            out_mlu_fp16 = traced_model(input_x_mlu.half())
            out_cpu_fp16 = model(input_x.half().float())
            self.assertTensorsEqual(out_cpu_fp16, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

            model2 = TestUpsampleModel2(scale_factor).eval()
            traced_model2 = torch.jit.trace(model2, input_x, check_trace=False)
            out_cpu2 = model2(input_x)
            out_mlu2 = traced_model2(input_x_mlu)
            self.assertTensorsEqual(out_cpu2, out_mlu2.cpu(), 0.003, use_MSE = True)
            # Test for fp16
            traced_model2.half()
            out_mlu_fp16 = traced_model2(input_x_mlu.half())
            out_cpu_fp16 = model2(input_x.half().float())
            self.assertTensorsEqual(out_cpu_fp16, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
