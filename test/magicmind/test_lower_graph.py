from __future__ import print_function
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
    traced_model = torch.jit.script(mlu_model)
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

def run_test(x, model, assert_func):
    mlu_out = run_mfus_forward(x, model)
    cpu_out = run_cpu_forward(x, model)
    assert_func(cpu_out, mlu_out.cpu(), 3e-3, use_MSE = True)

class TestExceptionPassOp(torch.nn.Module):
    def __init__(self):
        super(TestExceptionPassOp, self).__init__()
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = x + x
        if y.sum() < -1000:
            raise Exception("Sum should be greater than -1000")
        else:
            pass
        return y
class TestInplaceOp(torch.nn.Module):
    def __init__(self):
        super(TestInplaceOp, self).__init__()

    def forward(self, x):
        y = x + x
        r = y * 2 - x
        r.sub_(x)
        return r

class TestRemoveDropout(torch.nn.Module):
    def __init__(self):
        super(TestRemoveDropout,self).__init__()
        self.dp = nn.Dropout(0.5)

    def forward(self,x):
        y = x + x
        z = self.dp(y)*2
        return z.contiguous()

class TestLowerGraph(TestCase):
    @testinfo()
    #@unittest.skip("not test")
    def test_exception_pass_op(self):
        print("****Test Remove Exception/Pass Pattern in graph****")
        inputX = torch.rand(1,16,50,50)
        model = TestExceptionPassOp()
        run_test(inputX, model, self.assertTensorsEqual)

    @testinfo()
    #@unittest.skip("not test")
    def test_replace_inplace_op(self):
        print("****Test Replace Inplace ops for normal ops****")
        inputX = torch.rand(1,16,50,50)
        model = TestInplaceOp()
        run_test(inputX, model, self.assertTensorsEqual)

    @testinfo()
    #@unittest.skip("not test")
    def test_remove_dropout(self):
        print("****Test Remove Dropout****")
        inputX = torch.rand(1,4,10,10)
        model = TestRemoveDropout()
        run_test(inputX, model, self.assertTensorsEqual)

    #TODO: add conv test in the future.
if __name__ == '__main__':
    unittest.main()
