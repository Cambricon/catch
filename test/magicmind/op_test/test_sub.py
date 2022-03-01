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

class TestSubModel(nn.Module):
    def __init__(self, alpha = 1.0):
        super(TestSubModel, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        z = torch.sub(x, y, alpha = self.alpha)
        return z

class TestSubScalarModel(nn.Module):
    def __init__(self, scalar = 1.0, alpha = 1.0):
        super(TestSubScalarModel, self).__init__()
        self.scalar = scalar
        self.alpha = alpha

    def forward(self, x):
        y = torch.sub(x, self.scalar, alpha = self.alpha)
        return y

class TestSubOp(TestCase):
    def test_sub(self):
        # Test sub.Tensor
        alpha = random.random()
        model = TestSubModel(alpha)

        input_x = torch.rand(3, 5, 6)
        input_y = torch.rand(3, 5, 6)

        traced_model = torch.jit.trace(model, (input_x, input_y), check_trace=False)

        input_x_mlu = input_x.to('mlu')
        input_y_mlu = input_y.to('mlu')
        # Test for fp32 & fp16
        out_cpu = model(input_x, input_y)
        out_mlu = traced_model(input_x_mlu, input_y_mlu)
        out_mlu_fp16 = traced_model(input_x_mlu.half(), input_y_mlu.half())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE = True)
        self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.03, use_MSE = True)

        # Test sub.Scalar
        scalar = random.random()
        alpha = random.random()
        model = TestSubScalarModel(scalar, alpha)
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
