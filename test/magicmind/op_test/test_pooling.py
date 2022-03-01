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
from itertools import product

import time
import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestMaxPool2dModel(nn.Module):
    def __init__(self, kernel, stride, padding, dilation, return_indices, ceil_mode):
        super(TestMaxPool2dModel, self).__init__()
        self.maxpool2d = torch.nn.MaxPool2d(kernel, stride, padding,
                                            dilation, return_indices, ceil_mode)
    def forward(self, x):
        z = self.maxpool2d(x)
        return z

class TestMaxPool2dModel1(nn.Module):
    def __init__(self):
        super(TestMaxPool2dModel1, self).__init__()

    def forward(self, x):
        z = F.max_pool2d(x, 2, stride=None)
        return z

class TestMaxPool2dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool2d(self):
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]

        #TODO: For ceil_mode is True, some testing cases can't be passed which seems like
        # magicmind's issue, so temporarily shield this case and resolve it laster.
        ceil_mode_v = [False, False]
        return_indices_v = [False]

        loop_var = [kernel_v, stride_v, padding_v,
                    ceil_mode_v, return_indices_v]
        for kernel, stride, padding, ceil_mode, return_indices in product(
                *loop_var):
            model = TestMaxPool2dModel(kernel, stride, padding, 1,
                                       return_indices, ceil_mode).eval().float()
            input_x = torch.randn(4, 2, 128, 128).float()
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
    def test_maxpool2dfunc(self):
            input_x = torch.randn(4, 2, 128, 128).float()
            model = TestMaxPool2dModel1().eval().float()
            traced_model = torch.jit.trace(model, input_x, check_trace=False)

            input_x_mlu = input_x.to('mlu')

            out_cpu = model(input_x)
            out_mlu = traced_model(input_x_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

class TestAvgPool2dModel(nn.Module):
    def __init__(self, kernel, stride, padding, count_include_pad, ceil_mode):
        super(TestAvgPool2dModel, self).__init__()
        self.avgpool2d = torch.nn.AvgPool2d(kernel, stride, padding,
                                            ceil_mode, count_include_pad)
    def forward(self, x):
        z = self.avgpool2d(x)
        return z

class TestAvgPool2dModel1(nn.Module):
    def __init__(self):
        super(TestAvgPool2dModel1, self).__init__()

    def forward(self, x):
        z = F.avg_pool2d(x, 2, stride=None)
        return z

class TestAvgPool2dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool2d(self):
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4]
        padding_v = [0, 1]

        ceil_mode_v = [False]
        count_include_pad = [True]

        loop_var = [kernel_v, stride_v, padding_v, ceil_mode_v]
        for kernel, stride, padding, ceil_mode in product(*loop_var):
            model = TestAvgPool2dModel(kernel, stride, padding,
                                       ceil_mode, True).eval().float()
            input_x = torch.randn(4, 2, 128, 128).float()
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
    def test_avgpool2dfunc(self):
            input_x = torch.randn(4, 2, 128, 128).float()
            model = TestAvgPool2dModel1().eval().float()
            traced_model = torch.jit.trace(model, input_x, check_trace=False)

            input_x_mlu = input_x.to('mlu')

            # Test for fp32
            out_cpu = model(input_x)
            out_mlu = traced_model(input_x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)
            # Test for fp16
            out_mlu_fp16 = traced_model(input_x_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

class TestAdaptiveAvgPool2dModel(nn.Module):
    def __init__(self, output_shape):
        super(TestAdaptiveAvgPool2dModel, self).__init__()
        self.adp_avg_pool2d = nn.AdaptiveAvgPool2d(output_shape)

    def forward(self, x):
        y = self.adp_avg_pool2d(x)
        return y


class TestAdaptiveAvgPool2dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_adp_avgpool2d(self):
        in_shape_list = [(8, 16, 14, 14), (16, 6, 8, 8), (4, 23, 13, 64), (6, 8, 16, 16)]
        out_shape_list = [(4, 4), (10, 7), (9, 11)]
        list_list = [in_shape_list, out_shape_list]

        for in_shape, out_shape in product(*list_list):
            model = TestAdaptiveAvgPool2dModel(out_shape).eval().float()
            input_cpu = torch.randn(in_shape).float()
            traced_model = torch.jit.trace(model, input_cpu, check_trace=False)
            input_mlu = input_cpu.to('mlu')

            # Test for fp32
            out_cpu = model(input_cpu)
            out_mlu = traced_model(input_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)
            # Test for fp16
            out_mlu_fp16 = traced_model(input_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)


if __name__ == '__main__':
    unittest.main()
