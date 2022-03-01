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

class TestTranposeModel(nn.Module):
    def __init__(self, dims):
        super(TestTranposeModel, self).__init__()
        self.dims_ = dims

    def forward(self, x):
        z = torch.transpose(x, self.dims_[0], self.dims_[1])
        return z

class TestInplaceTranspose(nn.Module):
    def __init__(self, dims):
        super(TestInplaceTranspose, self).__init__()
        self.dims_ = dims

    def forward(self, x):
        x.transpose_(self.dims_[0], self.dims_[1])
        return x

class TestTransposeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_transpose(self):
        dims_list = ([1,3], [0,2])
        for in_shape in [(2,3,16,32)]:
            for dims in dims_list:
                model = TestTranposeModel(dims)

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
    def test_transpose_inplace(self):
        dims_list = ([1,3], [0,2])
        for in_shape in [(2,3,16,32)]:
            for dims in dims_list:
                model = TestInplaceTranspose(dims)

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

class TestTModel(nn.Module):
    def __init__(self):
        super(TestTModel, self).__init__()

    def forward(self, x):
        z = torch.t(x)
        return z

class TestInplaceT(nn.Module):
    def __init__(self):
        super(TestInplaceT, self).__init__()

    def forward(self, x):
        x.t_()
        return x

class TestTOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_t(self):
        for in_shape in [(3,6)]:
            model = TestTModel()

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
    def test_t_inplace(self):
        for in_shape in [(3,6)]:
            model = TestInplaceT()

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
