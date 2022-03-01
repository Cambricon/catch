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

class TestDivModel(nn.Module):
    def __init__(self):
        super(TestDivModel, self).__init__()

    def forward(self, x, y):
        z = torch.div(x, y)
        return z

class TestDivOp(TestCase):
    def test_div(self):
        # Test sub.Tensor
        alpha = random.random()
        model = TestDivModel()

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

if __name__ == '__main__':
    unittest.main()
