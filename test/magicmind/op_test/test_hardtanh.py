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
from itertools import product

import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestHardtanh(nn.Module):
    def __init__(self, inplace=True):
        super(TestHardtanh, self).__init__()
        self.layer = nn.Hardtanh(inplace=inplace, min_val=0., max_val=6.)

    def forward(self, x):
        y = self.layer(x)
        return y

class TestHardtanhOp(TestCase):
    @testinfo()
    # @unittest.skip("not test")
    def test_hardtanh(self):
        D = [1, 3]
        N = [1, 8]
        C = [1, 3]
        HW = [244]
        loop_val = [N, D, C, HW, HW]
        model = TestHardtanh(inplace=False).float().eval()
        model_ = TestHardtanh(inplace=True).float().eval()
        input_types = [torch.half, torch.float]
        for shape in product(*loop_val):
            for input_type in input_types:
                # no-inplace mode
                input = torch.randn(shape).to(dtype = input_type)
                if input_type == torch.half:
                    output_cpu = model(input.float())
                else:
                    output_cpu = model(input)
                traced = torch.jit.trace(model.to(ct.mlu_device()), self.to_mlu(input), check_trace=False)
                output_mlu = traced(self.to_mlu(input))
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE = True)

                # inplace mode
                input_cpu = torch.randn(shape).to(dtype = input_type)
                input_mlu = self.to_mlu(copy.deepcopy(input_cpu))
                if input_type == torch.half:
                    output_cpu = model(input_cpu.float())
                else:
                    output_cpu = model(input_cpu)

                traced = torch.jit.trace(model_.to(ct.mlu_device()), self.to_mlu(input_cpu), check_trace=False)
                output_mlu = traced(input_mlu)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
