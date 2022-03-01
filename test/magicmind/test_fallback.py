import os
os.environ["ENABLE_FALLBACK_TO_CPU"] = "1"

import torch
import torch.nn as nn
from torch.nn import Parameter
import unittest
import logging
import torch_mlu
import numpy as np
from torchvision import models

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import testinfo, TestCase
torch.set_grad_enabled(False)

# torch.mv, torch.rrelu are not supported
class TestMVCase(torch.nn.Module):
    def __init__(self):
        super(TestMVCase, self).__init__()

    def forward(self, mat, vec):
        x = torch.mv(mat, vec)
        x = x + x
        x = torch.rrelu(x)
        return x

class TestFallback(TestCase):
    @testinfo()
    # @unittest.skip("not test")
    def test_fallback2cpu_case(self):
        print("****fall back to cpu torch.mv****")
        model = TestMVCase().eval().float()
        mat = torch.randn(2, 3)
        vec = torch.randn(3)
        traced_model = torch.jit.trace(model, (mat, vec), check_trace=False)
        mat_mlu = mat.to('mlu')
        vec_mlu = vec.to('mlu')

        out_cpu = model(mat, vec)
        out_mlu = traced_model(mat_mlu, vec_mlu)

        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    @testinfo()
    # @unittest.skip("not test")
    def test_googlenet(self):
        print("****incepv2 fallback to cnnl****")
        input = torch.rand(1,3,299,299, dtype=torch.float)
        model = models.googlenet(pretrained=False).eval().float()
        out_cpu = model(input)
        script_module = torch.jit.trace(model, input, check_trace=False)
        script_module.to("mlu")
        out_mlu = script_module(input.to("mlu"))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    @testinfo()
    # @unittest.skip("not test")
    def test_inceptionV3(self):
        print("****incepv3 fallback to cnnl****")
        input = torch.rand(1,3,299,299, dtype=torch.float)
        model = models.inception_v3(pretrained=False).eval().float()
        out_cpu = model(input)
        script_module = torch.jit.trace(model, input, check_trace=False)
        script_module.to("mlu")
        out_mlu = script_module(input.to("mlu"))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
