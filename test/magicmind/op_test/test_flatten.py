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

import time
import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestFlattenModel(nn.Module):
    def __init__(self):
        super(TestFlattenModel, self).__init__()

    def forward(self, x):
        z = torch.flatten(x)
        return z

class TestNNFlattenModel(nn.Module):
    def __init__(self):
        super(TestNNFlattenModel, self).__init__()
        self.flatten = torch.nn.Flatten(0,-1)

    def forward(self, x):
        z = self.flatten(x)
        return z


class TestFlattenOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_flatten(self):
        for in_shape in [(1), (2, 3), (8, 224, 224), (1, 1, 1, 1), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            model = TestFlattenModel()

            input_x = torch.randn(in_shape)

            traced_model = torch.jit.trace(model, input_x, check_trace=False)

            input_x_mlu = input_x.to('mlu')

            # Test for fp32
            out_cpu = model(input_x)
            out_mlu = traced_model(input_x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)
            # Test for fp16
            out_mlu_fp16 = traced_model(input_x_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nn_flatten(self):
        for in_shape in [(8, 224, 224), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            model = TestNNFlattenModel()

            input_x = torch.randn(in_shape)

            traced_model = torch.jit.trace(model, input_x, check_trace=False)

            input_x_mlu = input_x.to('mlu')

            # Test for fp32
            out_cpu = model(input_x)
            out_mlu = traced_model(input_x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)
            # Test for fp16
            out_mlu_fp16 = traced_model(input_x_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
