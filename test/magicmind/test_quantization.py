from __future__ import print_function
import torch
import torch.nn as nn
import torch_mlu
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import sys
import os

import time
import unittest
import random
import copy
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import testinfo, TestCase
import torch_mlu.core.mlu_quantize as mlu_quantize

torch.set_grad_enabled(False)

class TestModel(nn.Module):
    def __init__(self, in_channels):
        super(TestModel, self).__init__()
        out_channels = 16
        conv1 = torch.nn.Conv2d(in_channels, out_channels, \
                                     3, 1, 0, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels, affine=False)
        self.relu1 = torch.nn.ReLU(inplace=False)

        self.block1 = torch.nn.Sequential(conv1,
                                          self.bn1,
                                          self.relu1)

    def forward(self, x):
        y = self.block1(x)
        z = y + y
        z1 = F.max_pool2d(z, 2, stride=None)
        z2 = torch.transpose(z1, 0, 3)
        return z2

class TestQuantizedModel(TestCase):
    @testinfo()
    def test_quantized_model(self):
        in_channels = random.randint(1,10)
        input1 = torch.rand(1, in_channels, 224, 224)
        net_cpu = TestModel(in_channels)
        net_cpu.eval().float()
        output_cpu = net_cpu(input1)

        quantized_net = mlu_quantize.quantize_dynamic_mlu(net_cpu,
                                                          {'use_avg': False,
                                                           'data_scale': 1.0,
                                                           'mean': None,
                                                           'std': None,
                                                           'firstconv': False,
                                                           'per_channel': True,
                                                           'asymettry': False},
                                                          dtype = "int8",
                                                          gen_quant = True)

        # quantization process
        output1 = quantized_net(input1)
        checkpoint = quantized_net.state_dict()

        # inference process
        infer_model = mlu_quantize.quantize_dynamic_mlu(net_cpu)
        infer_model.load_state_dict(checkpoint)
        traced_model = torch.jit.trace(infer_model.to('mlu'), input1.to('mlu'), check_trace=False)

        output_mlu = traced_model(input1.to('mlu'))
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.02, use_MSE=True)

if __name__ == '__main__':
    unittest.main()


