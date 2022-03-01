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

import time
import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestStackModel(nn.Module):
    def __init__(self, dim):
        super(TestStackModel, self).__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.stack([x,x,x], self.dim)
        return y

class TestStackMixModel(nn.Module):
    def __init__(self, dim, shape):
        super(TestStackMixModel, self).__init__()
        self.dim = dim
        self.other = torch.randn(shape)

    def forward(self, x):
        y = torch.stack([x,x,self.other], self.dim)
        return y

class TestStackOp(TestCase):
    # @unittest.skip("not test")
    def test_stack(self):
        shapes = [(2),(2,4),(2,2,4),(2,3,4,4),(2,4,2,2,3)]
        dims = [0, 1, 2, 3, 4,5]
        for shape in shapes:
            shape_len = 2 if isinstance(shape, int) else len(shape)+1
            for dim in range(0,shape_len):
                model = TestStackModel(dim).eval()
                input_x = torch.randn(shape).float()
                traced_model = torch.jit.trace(model, input_x, check_trace=False)
                out_cpu = model(input_x)

                input_x_mlu = input_x.to('mlu')
                out_mlu = traced_model(input_x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE = True)
                # Test for fp16
                traced_model.half()
                out_mlu_fp16 = traced_model(input_x_mlu.half())
                out_cpu_fp16 = model(input_x.half().float())
                self.assertTensorsEqual(out_cpu_fp16, out_mlu_fp16.cpu(), 0.0, use_MSE = True)


    # @unittest.skip("not test")
    def test_stack_const(self):
        dims = [0, 1, 2, 3]
        for dim in dims:
            model = TestStackMixModel(dim, (2,4,2)).eval()
            input_x = torch.randn((2,4,2)).float()
            traced_model = torch.jit.trace(model, input_x, check_trace=False)
            input_x_mlu = input_x.to('mlu')

            out_cpu = model(input_x)
            out_mlu = traced_model(input_x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
