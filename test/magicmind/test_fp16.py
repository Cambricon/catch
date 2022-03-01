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
torch.set_grad_enabled(False)

class FP16Model(nn.Module):
    def __init__(self, in_channels):
        super(FP16Model, self).__init__()
        out_channels = 16
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, \
                                     3, 1, 0, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels, affine=False)
        self.relu1 = torch.nn.ReLU(inplace=False)

        self.block1 = torch.nn.Sequential(self.conv1,
                                          self.bn1,
                                          self.relu1)

    def forward(self, x):
        y = self.block1(x)
        z = y + y
        z1 = F.max_pool2d(z, 2, stride=None)
        z2 = torch.transpose(z1, 0, 3)
        return z2

class TestFP16Model(TestCase):
    @testinfo()
    def test_fp16_model(self):
        in_channels = random.randint(1,10)
        input1 = torch.rand(1, in_channels, 224, 224)
        net_cpu = FP16Model(in_channels)
        net_cpu.eval().float()
        output = net_cpu(input1)

        net_mlu = copy.deepcopy(net_cpu)
        traced_model = torch.jit.trace(net_mlu, input1, check_trace=False)
        traced_model.half().to("mlu")

        output_mlu = traced_model(input1.half().to('mlu'))

        self.assertTensorsEqual(output, output_mlu.cpu(), 0.02, use_MSE=True)

if __name__ == '__main__':
    unittest.main()


