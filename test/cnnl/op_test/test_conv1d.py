from __future__ import print_function
import logging
import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
from itertools import product
import unittest

import torch
from torch import nn
import torch.autograd

import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase # pylint: disable=C0413,C0411

def to_mlu(tensor_cpu):
    return tensor_cpu.to(ct.mlu_device())

class TestConvOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_convtrans1d(self):
        bias_lst = [True, False]
        N_lst = [3]
        Ci_lst = [64, 32]
        HW_lst = [20, 11]
        Co_lst = [32, 16]
        K_lst = [1, 2]
        padding_lst = [1, 3]
        stride_lst = [1, 3]
        dilation_lst = [1, 2]
        outputpadding_lst = [0]
        groups_lst = [1, 2, 4]
        channel_func_lst = [self.convert_to_channel_last, lambda x:x]
        product_list = product(bias_lst,
                               N_lst,
                               Ci_lst,
                               HW_lst,
                               Co_lst,
                               K_lst,
                               padding_lst,
                               stride_lst,
                               dilation_lst,
                               outputpadding_lst,
                               groups_lst,
                               channel_func_lst)
        for bias_t, N, Ci, HW, Co, K, padding, stride, dilation, output_padding, \
                groups, channel_func in product_list:
            er = 0.003
            x = torch.randn(N, Ci, HW, dtype=torch.float, requires_grad=True)
            w = torch.randn(Ci, Co, K, dtype=torch.float, requires_grad=True)
            if bias_t:
                bias = torch.randn(Co * groups, dtype=torch.float, requires_grad=True)
            cm = nn.ConvTranspose1d(Ci, Co, K,
                                    stride=stride,
                                    padding=padding,
                                    output_padding=output_padding,
                                    bias=bias_t,
                                    dilation=dilation,
                                    groups = groups)
            cm.weight = torch.nn.Parameter(w)
            if bias_t:
                cm.bias = torch.nn.Parameter(bias)
            output_cpu = cm(x)
            grad_cpu = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x.grad)
            w_grad_cpu = copy.deepcopy(cm.weight.grad)
            if bias_t:
                bias_grad_cpu = copy.deepcopy(cm.bias.grad)
                cm.bias.grad.zero_()
            x.grad.zero_()
            cm.weight.grad.zero_()
            qcm = cm.to(ct.mlu_device())
            output_mlu = qcm(to_mlu(channel_func(x)))
            output_mlu.backward(to_mlu(channel_func(grad_cpu)))
            x_grad_mlu = x.grad.contiguous()
            w_grad_mlu = qcm.weight.grad.cpu().contiguous()
            if bias_t:
                bias_grad_mlu = qcm.bias.grad.cpu()
                self.assertTensorsEqual(bias_grad_cpu, bias_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu().contiguous(), er, use_MSE=True)
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv1d(self):
        bias_lst = [True, False]
        N_lst = [1, 32]
        Ci_lst = [128, 256]
        HW_lst = [11, 20]
        Co_lst = [64, 80, 128]
        K_lst = [1, 2]
        padding_lst = [0, 3]
        stride_lst = [1, 3]
        dilation_lst = [1, 2]
        product_list = product(bias_lst,
                               N_lst,
                               Ci_lst,
                               HW_lst,
                               Co_lst,
                               K_lst,
                               padding_lst,
                               stride_lst,
                               dilation_lst)
        for bias_t, N, Ci, HW, Co, K, padding, stride, dilation in product_list:
            er = 0.003
            x = torch.randn(N, Ci, HW, dtype=torch.float, requires_grad=True)
            cm = nn.Conv1d(Ci, Co, K,
                           bias=bias_t,
                           stride=stride,
                           padding=padding,
                           dilation=dilation)
            output_cpu = cm(x)
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad)
            x_grad_cpu = copy.deepcopy(x.grad)
            w_grad_cpu = copy.deepcopy(cm.weight.grad)
            if bias_t:
                bias_grad_cpu = copy.deepcopy(cm.bias.grad)
                cm.bias.grad.zero_()
            x.grad.zero_()
            cm.weight.grad.zero_()
            cm.to(ct.mlu_device())
            output_mlu = cm(to_mlu(x))
            output_mlu.backward(to_mlu(grad))
            x_grad_mlu = x.grad
            w_grad_mlu = cm.weight.grad.cpu()
            if bias_t:
                bias_grad_mlu = cm.bias.grad.cpu()
                self.assertTensorsEqual(bias_grad_cpu, bias_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), er, use_MSE=True)
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv1d_exceptions(self):
        x = torch.randn(15)
        cm = nn.Conv1d(3, 5, 2)
        cm.to('mlu')
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to('mlu'))
        msg = "Expected 3-dimensional input for 3-dimensional weight [5, 3, 2], " + \
              "but got 1-dimensional input of size [15] instead"
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(1,7,5)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to('mlu'))
        msg = "Given groups=1, weight of size [5, 3, 2], expected input[1, 7, 5] " +\
              "to have 3 channels, but got 7 channels instead"
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(10, 3, 5)
        x = x.to(torch.uint8)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to('mlu'))
        msg = "Convolution mlu op not implemented for 'Byte'"
        self.assertEqual(info.exception.args[0], msg)


if __name__ == "__main__":
    unittest.main()
