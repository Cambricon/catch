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
    def test_online_convtrans2d(self):
        bias_lst = [True, False]
        N_lst = [3]
        Ci_lst = [8]
        HW_lst = [14]
        Co_lst = [4]
        K_lst = [3]
        padding_lst = [0, 3]
        stride_lst = [2, 3]
        dilation_lst = [2, 3]
        outputpadding_lst = [0, 1]
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
            x = torch.randn(N, Ci, HW, HW, dtype=torch.float, requires_grad=True)
            w = torch.randn(Ci, Co, K, K, dtype=torch.float, requires_grad=True)
            if bias_t:
                bias = torch.randn(Co * groups, dtype=torch.float, requires_grad=True)
            cm = nn.ConvTranspose2d(Ci, Co, K,
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
    def test_online_conv2d(self):
        bias_lst = [True, False]
        N_lst = [32]
        Ci_lst = [3, 64]
        HW_lst = [14, 24]
        Co_lst = [64]
        K_lst = [2, 3]
        padding_lst = [0, 3]
        stride_lst = [1, 3]
        dilation_lst = [1]
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
            x = torch.randn(N, Ci, HW, HW, dtype=torch.float, requires_grad=True)
            w = torch.randn(Co, Ci, K, K, dtype=torch.float, requires_grad=True)
            if bias_t:
                bias = torch.randn(Co, dtype=torch.float, requires_grad=True)
            cm = nn.Conv2d(Ci, Co, K,
                           bias=bias_t,
                           stride=stride,
                           padding=padding,
                           dilation=dilation)
            cm.weight = torch.nn.Parameter(w)
            if bias_t:
                cm.bias = torch.nn.Parameter(bias)
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

    # Ref pytorch/test/test_nn.py:test_Conv2d_naive_groups,test_Conv2d_groups_nobias,
    #                             test_Conv2d_groups_nobias_v2
    # @unittest.skip("not test")
    @testinfo()
    def test_online_conv_groups(self):
        params_group = [
            [4, 4, 2, 2, False],
            [4, 4, 2, 2, True],
            [4, 16, 2, 8, False],
        ]
        for Ci, Co, ci, co, bias_t in params_group:
            i = torch.randn(2, Ci, 6, 6, requires_grad=True)
            w = torch.randn(Co, int(Ci/2), 3, 3, requires_grad=True)
            if bias_t:
                bias = torch.randn(Co, requires_grad=True)
            qcm = nn.Conv2d(Ci, Co, 3,
                         groups=2,
                         bias=bias_t).float().to(ct.mlu_device())
            qcm.weight = torch.nn.Parameter(w.to('mlu'))
            if bias_t:
                qcm.bias = torch.nn.Parameter(bias.to('mlu'))
            output = qcm(i.to('mlu'))
            grad_output = torch.randn(2, Co, 4, 4)
            output.backward(grad_output.to('mlu'))

            qcm1 = nn.Conv2d(ci, co, 3,
                          bias=bias_t).float().to(ct.mlu_device())
            qcm1.weight = torch.nn.Parameter(w[:co].to('mlu'))
            if bias_t:
                qcm1.bias = torch.nn.Parameter(bias[:co].to('mlu'))
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = qcm1(i1.to('mlu'))
            output1.backward(grad_output[:, :co].contiguous().to('mlu'))

            qcm2 = nn.Conv2d(ci, co, 3,
                          bias=bias_t).float().to(ct.mlu_device())
            qcm2.weight = torch.nn.Parameter(w[co:].to('mlu'))
            if bias_t:
                qcm2.bias = torch.nn.Parameter(bias[co:].to('mlu'))
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = qcm2(i2.to('mlu'))
            output2.backward(grad_output[:, co:].contiguous().to('mlu'))

            self.assertEqual(output.cpu(), torch.cat([output1.cpu(), output2.cpu()], 1))
            self.assertEqual(i.grad,
                             torch.cat([i1.grad, i2.grad], 1),
                             atol=1e-5, rtol=0)
            if bias_t:
                self.assertEqual(qcm.bias.grad.cpu(),
                                 torch.cat([qcm1.bias.grad.cpu(),
                                            qcm2.bias.grad.cpu()], 0),
                                 atol=1e-5, rtol=0)
            self.assertEqual(qcm.weight.grad.cpu(),
                             torch.cat([qcm1.weight.grad.cpu(),
                                        qcm2.weight.grad.cpu()], 0),
                             atol=1e-5, rtol=0)

    #@unittest.skip("not test")
    @testinfo()
    def test_depthwise_online_conv(self):
        N_lst = [1,8,32,64]
        Ci_lst = [3,16]
        HW_lst = [7,24]
        K_lst = [3]
        padding = [1, 1]
        stride = [1, 1]
        dilation = [1, 1]
        loop_var = [N_lst, Ci_lst, HW_lst, K_lst]
        for N, Ci, HW, K in product(*loop_var):
            m = (1,2,10)
            Cout = (Ci * x for x in m)
            for Co in Cout:
                err = 0.003
                x = torch.rand(N, Ci, HW, HW, dtype=torch.float, requires_grad=True)

                cm = nn.Conv2d(Ci, Co, K,
                               bias=True,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=Ci)
                output_cpu = cm(x)
                grad = torch.randn(output_cpu.shape, dtype=torch.float)
                output_cpu.backward(grad)
                x_grad_cpu = copy.deepcopy(x.grad)
                w_grad_cpu = copy.deepcopy(cm.weight.grad)
                bias_grad_cpu = copy.deepcopy(cm.bias.grad)

                x.grad.zero_()
                cm.weight.grad.zero_()
                cm.bias.grad.zero_()
                cm.to(ct.mlu_device())
                output_mlu = cm(to_mlu(x))
                output_mlu.backward(to_mlu(grad))
                x_grad_mlu = x.grad
                w_grad_mlu = cm.weight.grad.cpu()
                bias_grad_mlu = cm.bias.grad.cpu()
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), err, use_MSE=True)
                self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, err, use_MSE=True)
                self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, err, use_MSE=True)
                self.assertTensorsEqual(bias_grad_cpu, bias_grad_mlu, err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_online_depthwise_convtrans2d(self):
        bias_lst = [False, True]
        N_lst = [16]
        Ci_lst = [64]
        HW_lst = [32]
        Co_lst = [64]
        K_lst = [8]
        padding_lst = [2]
        stride_lst = [4]
        dilation_lst = [1]
        outputpadding_lst = [0]
        groups_lst = [64]
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
            x = torch.randn(N, Ci, HW, HW, dtype=torch.float, requires_grad=True)
            if bias_t:
                bias = torch.randn(Co, dtype=torch.float, requires_grad=True)
            cm = nn.ConvTranspose2d(Ci, Co, K,
                                    stride=stride,
                                    padding=padding,
                                    output_padding=output_padding,
                                    bias=bias_t,
                                    dilation=dilation,
                                    groups = groups)
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
    def test_conv2d_exceptions(self):
        x = torch.randn(15)
        cm = nn.Conv2d(3, 5, 2)
        cm.to('mlu')
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to('mlu'))
        msg = "Expected 4-dimensional input for 4-dimensional weight [5, 3, 2, 2], " + \
              "but got 1-dimensional input of size [15] instead"
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(1, 7, 5, 3)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to('mlu'))
        msg = "Given groups=1, weight of size [5, 3, 2, 2], expected input[1, 7, 5, 3] " +\
              "to have 3 channels, but got 7 channels instead"
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(10, 3, 5, 5)
        x = x.to(torch.int)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to('mlu'))
        msg = "Convolution mlu op not implemented for 'Int'"
        self.assertEqual(info.exception.args[0], msg)

if __name__ == "__main__":
    unittest.main()
