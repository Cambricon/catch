import torch
import torch.nn as nn
from torch.nn import Parameter
import unittest
import os
import logging
import torch_mlu
import numpy as np
from torchvision import models

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import testinfo, TestCase
torch.set_grad_enabled(False)

def run_mfus_forward(x, model):
    x_mlu = x.to("mlu")
    mlu_model = model.to("mlu")
    mlu_model.eval().float()
    traced_model = torch.jit.trace(mlu_model, x_mlu, check_trace=False)
    return traced_model(x_mlu)

def run_mfus_forward2(x, y, model):
    x_mlu = x.to("mlu")
    y_mlu = y.to("mlu")
    mlu_model = model.to("mlu")
    mlu_model.eval().float()
    traced_model = torch.jit.trace(mlu_model, {x_mlu, y_mlu}, check_trace=False)
    return traced_model(x_mlu, y_mlu)

def run_cpu_forward(x, model):
    model.eval().float()
    # traced_model = torch.jit.trace(model, x, check_trace=False)
    return model(x)
def run_cpu_forward2(x, y, model):
    model.eval().float()
    # traced_model = torch.jit.trace(model, x, check_trace=False)
    return model(x,y)
# x   x
#  \ /
#   add
#    |
#   relu
#     |
#   maxpool   1
#        |   /
#          x   4
#            \ /
#             sub
class TestModuleCase1(torch.nn.Module):
    def __init__(self):
        super(TestModuleCase1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x + x
        x = self.relu(x)
        x = self.maxpool(x)
        x = x + 1.
        x = x - x
        x = x - 4.
        return x

# x  x
# \  /
#  add
#   |
#   x     y
#   | \  /
#  add  sub
#   |    |
#   z    r
#    \   /
#     sub
class TestModuleCasea(torch.nn.Module):
    def __init__(self):
        super(TestModuleCasea, self).__init__()

    def forward(self, x, y):
        x = x + x
        z = x + x
        r = x - y
        return r - z

from torch.nn import functional as F
from torch.nn.parameter import Parameter

class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        self.conv.weight = Parameter(torch.ones_like(self.conv.weight))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        # self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        # branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class Conv_bn_relu(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Conv_bn_relu, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        return branch1x1

# relu
#  /    \
#conv1  conv2
# order conv2,conv1,relu, aliasDB move conv1 after conv2
class TestAB1B2(nn.Module):
    def __init__(self):
        super(TestAB1B2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(48, 64, bias=False,kernel_size=5, padding=2)
    
    def forward(self, x):
        y = self.relu(x)
        z = self.conv(y)
        p = self.conv(y)
        return (p, z)

class TestGraphSegment(TestCase):
    @testinfo()
    # @unittest.skip("not test")
    def test_pass_off(self):
        print("****graph partition off****")
        inputX = torch.rand(1,16,50,50)
        torch_mlu._MLUC._jit_override_can_fuse_on_mlu(False)
        model = TestModuleCase1().eval().float()
        traced_model = torch.jit.trace(model, inputX, check_trace=False)
        out_cnnl = traced_model(inputX.to("mlu"))
        torch_mlu._MLUC._jit_override_can_fuse_on_mlu(True)
        
    @testinfo()
    # @unittest.skip("not test")
    def test_partition_case1(self):
        print("****Test debug run on cpu****")
        inputX = torch.rand(1,16,50,50)
        model = TestModuleCase1()
        model_cpu = TestModuleCase1()
        os.environ["FUSED_KERNEL_DEBUG"] = "cpu"
        mlu_out = run_mfus_forward(inputX, model)
        cpu_out = run_cpu_forward(inputX, model_cpu)
        self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 3e-3, use_MSE = True)
        assert os.path.exists("mmouttensor_MLUFusionGroup0_0") and \
            os.path.exists("cpu_jitouttensor_MLUFusionGroup0_0"), \
            "FUSED_KERNEL_DEBUG=cpu has problem"
        del os.environ['FUSED_KERNEL_DEBUG']
        os.remove("mmouttensor_MLUFusionGroup0_0")
        os.remove("cpu_jitouttensor_MLUFusionGroup0_0")

    @testinfo()
    # @unittest.skip("not test")
    def test_parallel_case(self):
        print("****graph partition parallel****")
        model = TestAB1B2().eval().float()
        input_x = torch.randn(1, 48, 224, 224)
        traced_model = torch.jit.trace(model, input_x, check_trace=False)
        traced_model.to("mlu")
        input_x_mlu = input_x.to('mlu')

        out_cpu1, out_cpu2 = model(input_x)
        out_mlu1, out_mlu2 = traced_model(input_x_mlu)

        self.assertTensorsEqual(out_cpu1, out_mlu1.cpu(), 0.003, use_MSE = True)
        self.assertTensorsEqual(out_cpu2, out_mlu2.cpu(), 0.003, use_MSE = True)

    @testinfo()
    # @unittest.skip("not test")
    def test_Conv_bn_relu(self):
        # torch.manual_seed(6)
        print("****conv_bn_relu fallback to cnnl****")
        input = torch.rand(1, 4, 2, 2, dtype=torch.float)
        model = Conv_bn_relu(4, 4).eval().float()
        out_cpu = model(input)
        traced_model = torch.jit.trace(model, input, check_trace=False)
        traced_model.to("mlu")
        
        out_mlu = traced_model(input.to("mlu"))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    @testinfo()
    # @unittest.skip("not test")
    def test_inceptionA(self):
        print("****inception block fallback to cnnl****")
        input = torch.rand(1, 64, 224, 224, dtype=torch.float)
        model = InceptionA(64, 64).eval().float()
        out_cpu = model(input)
        traced_model = torch.jit.trace(model, input, check_trace=False)
        traced_model.to("mlu")        
        out_mlu = traced_model(input.to("mlu"))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    @testinfo()
    # @unittest.skip("not test")
    def test_partition_fallback_blacklist(self):
        print("****test_fallback_blacklist****")
        inputX = torch.rand(1,2,3,3, dtype=torch.float32)
        inputY = torch.rand(1,2,3,3, dtype=torch.float32)
        model = TestModuleCasea()
        model_cpu = TestModuleCasea()
        os.environ["DEBUG_FORCED_FALLBACK_OPS"] = "aten::add.Tensor,aten::sub.Tensor"
        mlu_out = run_mfus_forward2(inputX, inputY, model)
        del os.environ['DEBUG_FORCED_FALLBACK_OPS']
        mlu_out = run_mfus_forward2(inputX, inputY, model)       

if __name__ == '__main__':
    unittest.main()
