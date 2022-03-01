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
import torch_mlu.core.mlu_quantize as mlu_quantize

import time
import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestAddMMModel(nn.Module):
    def __init__(self, beta=1.0, alpha=2.0):
        super(TestAddMMModel, self).__init__()
        self.beta_  = beta
        self.alpha_ = alpha

    def forward(self, i, x, y):
        z = torch.addmm(i, x, y, beta=self.beta_, alpha=self.alpha_)
        return z

class TestAddMMInplace(nn.Module):
    def __init__(self, beta=1.0, alpha=2.0):
        super(TestAddMMInplace, self).__init__()
        self.beta_  = beta
        self.alpha_ = alpha

    def forward(self, i, x, y):
        i.addmm_(x, y, beta=self.beta_, alpha=self.alpha_)
        return i

class TestQuantizedLinearModel(nn.Module):
    def __init__(self, in_chl_, out_chl_, bias):
        super(TestQuantizedLinearModel, self).__init__()
        self.linear = nn.Linear(in_chl_, out_chl_, bias)

    def forward(self, x):
        return self.linear(x)

class TestAddMMOp(TestCase):
    # @unittest.skip("not test")
    def test_addmm(self):
        model = TestAddMMModel(2.0, 1.0)

        i_shapes = [(1), (3,9), (10,20)]
        x_shapes = [(2,4), (3,4), (10,30)]
        y_shapes = [(4,6), (4,9), (30,20)]

        ixy_shapes = [i_shapes, x_shapes, y_shapes]

        for i_shape, x_shape, y_shape in zip(*ixy_shapes):

            input_i = torch.randn(i_shape)
            input_x = torch.randn(x_shape)
            input_y = torch.randn(y_shape)

            traced_model = torch.jit.trace(model, (input_i, input_x, input_y), check_trace=False)

            input_i_mlu = input_i.to('mlu')
            input_x_mlu = input_x.to('mlu')
            input_y_mlu = input_y.to('mlu')

            # Test for fp32
            out_cpu = model(input_i, input_x, input_y)
            out_mlu = traced_model(input_i_mlu, input_x_mlu, input_y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.02, use_MSE = True)
            # Test for fp16
            out_mlu_fp16 = traced_model(input_i_mlu.half(), input_x_mlu.half(), input_y_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.02, use_MSE = True)

class TestAddMMInplaceOp(TestCase):
    # @unittest.skip("not test")
    def test_addmm_inplace(self):
        model = TestAddMMInplace(2.0, 1.0)

        # addmm_ not support 1-Dim
        i_shapes = [(2,6), (3,9), (10,20)]
        x_shapes = [(2,4), (3,4), (10,30)]
        y_shapes = [(4,6), (4,9), (30,20)]

        ixy_shapes = [i_shapes, x_shapes, y_shapes]

        for i_shape, x_shape, y_shape in zip(*ixy_shapes):

            input_i = torch.randn(i_shape)
            input_x = torch.randn(x_shape)
            input_y = torch.randn(y_shape)

            traced_model = torch.jit.trace(model, (input_i, input_x, input_y), check_trace=False)

            input_i_mlu = input_i.to('mlu')
            input_x_mlu = input_x.to('mlu')
            input_y_mlu = input_y.to('mlu')

            # Test for fp32
            out_cpu = model(input_i, input_x, input_y)
            out_mlu = traced_model(input_i_mlu, input_x_mlu, input_y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.02, use_MSE = True)
            # Test for fp16
            out_mlu_fp16 = traced_model(input_i_mlu.half(), input_x_mlu.half(), input_y_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.02, use_MSE = True)

class TestQLinearOp(TestCase):
    # @unittest.skip("not test")
    def test_linear_qint(self):
        bias_lst = [True, False]
        Ni_lst = [1, 10, 20]
        Ci_lst = [10, 50, 100]
        Co_lst = [25, 75, 125]
        bitwidth_lst = ['int8', 'int16']
        per_channel_lst = [True, False]
        half_input_lst = [0, 1]
        product_list = product(bias_lst,
                               Ni_lst,
                               Ci_lst,
                               Co_lst,
                               bitwidth_lst,
                               per_channel_lst,
                               half_input_lst)
        for bias, ni, ci, co, bw, per_channel, half_input in product_list:
            # model preparing
            model = TestQuantizedLinearModel(ci, co, bias)
            qconfig = {'use_avg': False,
                       'data_scale': 1.0,
                       'per_channel': per_channel,
                       'firstconv': False}

            quantized_net = mlu_quantize.quantize_dynamic_mlu(
                 model, qconfig, dtype = bw, gen_quant = True)

            # cpu forward
            input_t = torch.rand(ni,ci)
            output_cpu = model(input_t)

            # quantization process
            tmp = quantized_net(input_t)
            checkpoint = quantized_net.state_dict()


            # mlu forward
            model_mlu = mlu_quantize.quantize_dynamic_mlu(model)
            model_mlu.load_state_dict(checkpoint)
            traced_model = torch.jit.trace(
                model_mlu.to('mlu'), input_t.to('mlu'), check_trace = False)
            if per_channel:
                if half_input == 1:
                    model_mlu.half().to('mlu')
                    output_cnnl = model_mlu(input_t.half().to('mlu'))
                else:
                    model_mlu.to('mlu')
                    output_cnnl = model_mlu(input_t.to('mlu'))
                self.assertTensorsEqual(output_cpu, output_cnnl.cpu(), 0.02, use_MSE=True)
            else:
                if half_input == 1:
                    traced_model.half()
                    model_mlu.half().to('mlu')
                    output_mlu = traced_model(input_t.half().to('mlu'))
                    output_cnnl = model_mlu(input_t.half().to('mlu'))
                else:
                    output_mlu = traced_model(input_t.to('mlu'))
                    output_cnnl = model_mlu(input_t.to('mlu'))
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.02, use_MSE=True)
                self.assertTensorsEqual(output_cpu, output_cnnl.cpu(), 0.02, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
