from __future__ import print_function
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

class TestAddModel(nn.Module):
    def __init__(self, alpha = 1.0):
        super(TestAddModel, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        z = torch.add(x, y, alpha = self.alpha)
        return z

class TestAddScalarModel1(nn.Module):
    def __init__(self, scalar = 1.0, alpha = 1.0):
        super(TestAddScalarModel1, self).__init__()
        self.scalar = scalar
        self.alpha = alpha

    def forward(self, x):
        y = torch.add(x, self.scalar, alpha = self.alpha)
        return y

class TestAddScalarModel2(nn.Module):
    def __init__(self, scalar = 1.0):
        super(TestAddScalarModel2, self).__init__()
        self.scalar = scalar

    def forward(self, x):
        y = self.scalar + x
        return y

class TestAddOp(TestCase):
    def test_add(self):
        # Test add.Tensor
        alpha = random.random()
        model = TestAddModel(alpha)

        input_x = torch.rand(3, 5, 6)
        input_y = torch.rand(3, 5, 6)

        traced_model = torch.jit.trace(model, (input_x, input_y), check_trace=False)

        input_x_mlu = input_x.to('mlu')
        input_y_mlu = input_y.to('mlu')
        # Test for fp32 & fp16
        out_cpu = model(input_x, input_y)
        out_mlu = traced_model(input_x_mlu, input_y_mlu)
        out_mlu_fp16 = traced_model(input_x_mlu.half(), input_y_mlu.half())
        out_mlu_fp32 = traced_model(input_x_mlu.float(), input_y_mlu.half())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE = True)
        self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.03, use_MSE = True)
        self.assertTensorsEqual(out_cpu, out_mlu_fp32.cpu(), 0.03, use_MSE = True)

    def test_tensor_add_scalar(self):
        # Test Tensor + Scalar
        scalar = random.random()
        alpha = random.random()
        model = TestAddScalarModel1(scalar, alpha)
        input_x = torch.randn(1,3,4,4)
        traced_model = torch.jit.trace(model, input_x, check_trace=False)
        input_x_mlu = input_x.to('mlu')
        # Test for fp32 & fp16
        out_cpu = model(input_x)
        out_mlu = traced_model(input_x_mlu)
        out_mlu_fp16 = traced_model(input_x_mlu.half())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE = True)
        self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.03, use_MSE = True)

    def test_scalar_add_tensor(self):
        # Test Scalar + Tensor
        scalar = random.random()
        model = TestAddScalarModel2(scalar)
        input_x = torch.randn(1,3,4,4)
        traced_model = torch.jit.trace(model, input_x, check_trace=False)
        input_x_mlu = input_x.to('mlu')
        # Test for fp32 & fp16
        out_cpu = model(input_x)
        out_mlu = traced_model(input_x_mlu)
        out_mlu_fp16 = traced_model(input_x_mlu.half())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE = True)
        self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.03, use_MSE = True)


if __name__ == '__main__':
    unittest.main()
