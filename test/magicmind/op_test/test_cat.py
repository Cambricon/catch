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

class TestCatModel(nn.Module):
    def __init__(self, dim):
        super(TestCatModel, self).__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.cat([x,x], self.dim)
        return y

class TestCatOp(TestCase):
    # @unittest.skip("not test")
    def test_cat(self):
        ixy_shapes = [((-1),(1,4,2,2)),
                      ((0),(1,4,2,2)),
                      ((1),(1,4,2,2)),
                      ((2),(1,4,2,2)),
                      ((3),(1,4,2,2)),
                      ((-1),(1,4,2,2,3)),
                      ((0),(1,4,2,2,3)),
                      ((1),(1,4,2,2,3)),
                      ((2),(1,4,2,2,3)),
                      ((3),(1,4,2,2,3)),
                      ((-2),(1,2,3,4,5,6)),
                      ((4),(1,4,2,2,3))]

        for dim, x_shape in ixy_shapes:

            model = TestCatModel(dim).float().eval()

            input_x = torch.randn(x_shape)

            traced_model = torch.jit.trace(model, input_x, check_trace=False)

            input_x_mlu = input_x.to('mlu')

            # Test for fp32
            out_cpu = model(input_x)
            out_mlu = traced_model(input_x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

            # Test for fp16
            traced_model.half()
            out_mlu_fp16 = traced_model(input_x_mlu.half())
            out_cpu_fp16 = model(input_x.half().float())
            self.assertTensorsEqual(out_cpu_fp16, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
