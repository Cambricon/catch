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

class TestBatchNorm2dModel(nn.Module):
    def __init__(self, num_features, weight = None, bias = None, running_mean = None, running_var = None):
        super(TestBatchNorm2dModel, self).__init__()
        self.bn = torch.nn.BatchNorm2d(num_features, affine=False)
        if weight is not None:
            self.bn.weight = nn.Parameter(weight)
        if bias is not None:
            self.bn.bias = nn.Parameter(bias)
        if running_mean is not None:
            self.bn.running_mean = running_mean#nn.Parameter(running_mean)
        if running_var is not None:
            self.bn.running_var = running_var#nn.Parameter(running_var)

    def forward(self, x):
        z = self.bn(x)
        return z

class TestBatchNorm3dModel(nn.Module):
    def __init__(self, num_features, weight = None, bias = None, running_mean = None, running_var = None,):
        super(TestBatchNorm3dModel, self).__init__()
        self.bn = torch.nn.BatchNorm3d(num_features, affine=False)

        if weight is not None:
            self.bn.weight = nn.Parameter(weight)
        if bias is not None:
            self.bn.bias = nn.Parameter(bias)
        if running_mean is not None:
            self.bn.running_mean = running_mean
        if running_var is not None:
            self.bn.running_var =running_var

    def forward(self, x):
        z = self.bn(x)
        return z


class TestBatchNormdOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_fp32_fp16_datatype(self):
        N_lst = [3]
        Ci_lst = [3, 64]
        HW_lst = [14, 24]
        D_lst = [4, 8]
        Param_lst = [True, False]
        product_list = product(N_lst,
                               Ci_lst,
                               HW_lst,
                               Param_lst)
        for  N, Ci, HW, Param in product_list:
            if Param:
                weight = torch.rand(Ci).float()
                bias = torch.rand(Ci).float()
                rm = torch.rand(Ci).float()
                rv = torch.rand(Ci).float() + 1.0
            else:
                weight = None
                bias = None
                rm = None
                rv = None
            model = TestBatchNorm2dModel(Ci,
                                         weight = weight,
                                         bias = bias,
                                         running_mean = rm,
                                         running_var = rv).eval().float()
            input_t = torch.randn(N, Ci, HW, HW).float()
            traced_model = torch.jit.trace(model, input_t, check_trace=False)

            input_t_mlu = input_t.to('mlu')

            # Test for fp32
            out_cpu = model(input_t)
            out_mlu = traced_model(input_t_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

            # Test for fp16
            traced_model.half()
            out_mlu_fp16 = traced_model(input_t_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

        product_list_3d = product(N_lst,
                                  Ci_lst,
                                  D_lst,
                                  HW_lst,
                                  Param_lst)
        for N, Ci, D, HW, Param in product_list_3d:
            if Param:
                weight = torch.rand(Ci).float()
                bias = torch.rand(Ci).float()
                rm = torch.rand(Ci).float()
                rv = torch.rand(Ci).float() + 1.0
            else:
                weight = None
                bias = None
                rm = None
                rv = None
            model = TestBatchNorm3dModel(Ci,
                                         weight = weight,
                                         bias = bias,
                                         running_mean = rm,
                                         running_var = rv).eval().float()
            input_t = torch.randn(N, Ci, D, HW, HW).float()
            traced_model = torch.jit.trace(model, input_t, check_trace=False)

            input_t_mlu = input_t.to('mlu')

            # Test for fp32
            out_cpu = model(input_t)
            out_mlu = traced_model(input_t_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

            # Test for fp16
            traced_model.half()
            out_mlu_fp16 = traced_model(input_t_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)


if __name__ == '__main__':
    unittest.main()
