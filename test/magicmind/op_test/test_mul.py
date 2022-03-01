from __future__ import print_function
import torch
import torch.nn as nn
import torch_mlu
import torch_mlu.core.mlu_model as ct
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

class TestMulModel(nn.Module):
    def __init__(self):
        super(TestMulModel, self).__init__()

    def forward(self, x, y):
        z = x * y
        w = 1.5 * z
        return w

class TestMulCastModel(nn.Module):
    def __init__(self, x):
        super(TestMulCastModel, self).__init__()
        self.x = x

    def forward(self, y):
        return self.x * y + y * self.x

class TestMulScalarModel(nn.Module):
    def __init__(self, scalar = 1.0):
        super(TestMulScalarModel, self).__init__()
        self.scalar = scalar

    def forward(self, x):
        y = torch.mul(x, self.scalar)
        return y

class TestMulOp(TestCase):
    # @unittest.skip("not test")
    def test_mul(self):
        ixy_shapes = [((2,3,4,5,6),(2,3,4,5,6)),
                      ((1,3,224,224),(1,3,224,224)),
                      ((1,3,224,224),(1,3,1,1)),
                      ((1,1,24,1),(1,1,24,1)),
                      ((2,3,4),(2,3,4)),
                      ((2,3),(2,3)),
                      ((10),(1)),
                      ((1,3,224,1),(1,3,1,224)),
                      ((1,3,224,224),(1,1,1,1)),
                      ((1,3,224,224),(224))]

        for x_shape, y_shape in ixy_shapes:

            model = TestMulModel().float().eval()

            input_x = torch.randn(x_shape)
            input_y = torch.randn(y_shape)

            traced_model = torch.jit.trace(model, (input_x, input_y), check_trace=False)

            input_x_mlu = input_x.to('mlu')
            input_y_mlu = input_y.to('mlu')

            # Test for fp32
            out_cpu = model(input_x, input_y)
            out_mlu = traced_model(input_x_mlu, input_y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)
            # Test for fp16
            traced_model.half()
            out_mlu_fp16 = traced_model(input_x_mlu.half(), input_y_mlu.half())
            out_cpu_fp16 = model(input_x.half().float(), input_y.half().float())
            self.assertTensorsEqual(out_cpu_fp16, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    def test_mul_cast(self):
        x = torch.randn(1,3,224,224).float()
        y = torch.randn((1,3,224,224),dtype=torch.half)
        model_ = TestMulCastModel(x).float().eval()
        out_cpu = model_(y)

        x_mlu = x.to('mlu')
        y_mlu = y.to('mlu')
        model_mlu = TestMulCastModel(x).float().eval()

        traced = torch.jit.trace(model_, y, check_trace=False).to('mlu')

        out_fusion = traced(y.to('mlu'))
        self.assertTensorsEqual(out_cpu, out_fusion.cpu(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    def test_tensor_mul_scalar(self):
        # Test Tensor + Scalar
        scalar = random.random()
        model = TestMulScalarModel(scalar)
        input_x = torch.randn(1,3,4,4)
        traced_model = torch.jit.trace(model, input_x, check_trace=False)
        input_x_mlu = input_x.to('mlu')
        # Test for fp32 & fp16
        out_cpu = model(input_x)
        out_mlu = traced_model(input_x_mlu)
        out_mlu_fp16 = traced_model(input_x_mlu.half())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)
        self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
