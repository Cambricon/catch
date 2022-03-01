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

import time
import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import torch_mlu.core.mlu_quantize as mlu_quantize
import logging

from torchvision import transforms
from PIL import Image
import random

logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestConv2dModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias):
        super(TestConv2dModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)

    def forward(self, x):
        z = self.conv1(x)
        return z

class TestConv2dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_conv2d_fp32_fp16_datatype(self):
        bias_lst = [True, False]
        N_lst = [3]
        Ci_lst = [3, 64]
        HW_lst = [14, 24]
        Co_lst = [64]
        K_lst = [2, 3]
        padding_lst = [0, 3]
        stride_lst = [1, 3]
        dilation_lst = [1]
        groups_lst = [1]
        product_list = product(bias_lst,
                               N_lst,
                               Ci_lst,
                               HW_lst,
                               Co_lst,
                               K_lst,
                               padding_lst,
                               stride_lst,
                               dilation_lst,
                               groups_lst)
        for bias_t, N, Ci, HW, Co, K, padding, stride, dilation, groups in product_list:
            model = TestConv2dModel(Ci, Co, K, stride, padding,
                                    dilation, groups, bias_t).eval().float()
            input_t = torch.randn(N, Ci, HW, HW).float()
            traced_model = torch.jit.trace(model, input_t, check_trace=False)

            input_t_mlu = input_t.to('mlu')

            # Test for fp32
            out_cpu = model(input_t)
            out_mlu = traced_model(input_t_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

            # Test for fp16
            traced_model.half()
            out_mlu_fp16= traced_model(input_t_mlu.half())
            self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_groups_conv2d(self):
        model = TestConv2dModel(4, 16, (3,3), (1,1), (0,0), (1,1), 2, False).eval().float()

        input_t = torch.randn(2, 4, 6, 6).float()
        traced_model = torch.jit.trace(model, input_t, check_trace=False)

        input_t_mlu = input_t.to('mlu')

        # Test for fp32
        out_cpu = model(input_t)
        out_mlu = traced_model(input_t_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

        # Test for fp16
        traced_model.half()
        out_mlu_fp16 = traced_model(input_t_mlu.half())
        self.assertTensorsEqual(out_cpu, out_mlu_fp16.cpu(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv2d_qint(self):
        bias_lst = [True, False]
        N_lst = [3]
        Ci_lst = [64]
        HW_lst = [24]
        Co_lst = [64]
        K_lst = [3]
        padding_lst = [3]
        stride_lst = [3]
        dilation_lst = [1]
        groups_lst = [1]
        bitwidth_lst = ['int8', 'int16']
        per_channel_lst = [True, False]
        half_input_lst = [0, 1]
        product_list = product(bias_lst,
                               N_lst,
                               Ci_lst,
                               HW_lst,
                               Co_lst,
                               K_lst,
                               padding_lst,
                               stride_lst,
                               dilation_lst,
                               groups_lst,
                               bitwidth_lst,
                               per_channel_lst,
                               half_input_lst)
        for bias_t, N, Ci, HW, Co, K, padding, stride, dilation, groups, bw, per_channel, half_input in product_list:
            model = TestConv2dModel(Ci, Co, K, stride, padding, dilation, groups, bias_t).eval().float()
            qconfig = {'use_avg': False, 'data_scale': 1.0, 'per_channel': per_channel, 'firstconv': False,}
            quantized_net = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype = bw, gen_quant = True)

            # cpu forward
            input_t = torch.randn(N, Ci, HW, HW).float()
            output_cpu = model(input_t)

            # quantization process
            tmp = quantized_net(input_t)
            checkpoint = quantized_net.state_dict()

            # mlu forward
            model_mlu = mlu_quantize.quantize_dynamic_mlu(model)
            model_mlu.load_state_dict(checkpoint)
            traced_model = torch.jit.trace(model_mlu.to('mlu'), input_t.to('mlu'), check_trace = False)
            if half_input == 1:
                traced_model.half()
                model_mlu.half()
                output_mlu = traced_model(input_t.half().to('mlu'))
                output_cnnl = model_mlu(input_t.half().to('mlu'))
            else:
                output_mlu = traced_model(input_t.to('mlu'))
                output_cnnl = model_mlu(input_t.to('mlu'))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.02, use_MSE=True)
            self.assertTensorsEqual(output_cpu, output_cnnl.cpu(), 0.02, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv2d_first_qint(self):
        bitwidth_lst = ['int8']
        per_channel_lst = [True]
        half_input_lst = [1]
        in_chl_lst = [4]
        product_list = product(bitwidth_lst,
                               per_channel_lst,
                               half_input_lst,
                               in_chl_lst,)
        for bw, per_channel, half_input, in_chl_ in product_list:
            out_chl_ = random.randint(1, 100)
            h_shape_ = random.randint(100, 1000)
            w_shape_ = random.randint(100, 1000)
            kernel_size_ = random.randint(1,10)

            # 1. prepare a random img
            np_array = np.random.randint(0, 255, in_chl_ * h_shape_ * w_shape_).reshape([h_shape_,w_shape_,in_chl_])
            img_mode = 'RGB' if in_chl_ == 3 else 'RGBA'
            imat = Image.fromarray(np_array.astype('uint8'), img_mode)
            # 2. prepare input for cpu
            if in_chl_ == 3:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                mean_mlu = [0, 0, 0]
                std_mlu = [1/255, 1/255, 1/255]
            else:
                mean = [0.485, 0.456, 0.406, 0.449]
                std = [0.229, 0.224, 0.225, 0.226]
                mean_mlu = [0, 0, 0, 0]
                std_mlu = [1/255, 1/255, 1/255, 1/255]

            transform_cpu = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            inputs_cpu = transform_cpu(imat).unsqueeze(0)

            # 3. prepare input for mlu
            transform_mlu = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_mlu, std=std_mlu)])
            inputs_mlu = transform_mlu(imat).unsqueeze(0)
            # 4. prepare conv model and run cpu forward
            stride_v = [1,4]
            padding_v = [2,2]
            dilation_v = [1,1]
            groups = 1
            model = TestConv2dModel(in_chl_, out_chl_, kernel_size_, stride_v, padding_v, dilation_v, groups, True).eval().float()
            output_cpu = model(inputs_cpu)

            # 5. do quantization
            qconfig = {'use_avg': False,
                       'data_scale': 1.0,
                       'mean': mean,
                       'std': std,
                       'firstconv': True,
                       'per_channel': per_channel}
            quantized_net = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype = bw, gen_quant = True)
            tmp = quantized_net(inputs_cpu)
            checkpoint = quantized_net.state_dict()

            # 6. run mlu forward
            model_mlu = mlu_quantize.quantize_dynamic_mlu(model)
            model_mlu.load_state_dict(checkpoint)
            traced_model = torch.jit.trace(model_mlu.to('mlu'), inputs_mlu.to('mlu'), check_trace = False)
            if half_input == 1:
                traced_model.half()
                model_mlu.half()
                output_mlu = traced_model(inputs_mlu.half().to('mlu'))
                output_cnnl = model_mlu(inputs_mlu.half().to('mlu'))
            else:
                output_mlu = traced_model(inputs_mlu.to('mlu'))
                output_cnnl = model_mlu(inputs_mlu.to('mlu'))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.02, use_MSE=True)
            self.assertTensorsEqual(output_cpu, output_cnnl.cpu(), 0.02, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
