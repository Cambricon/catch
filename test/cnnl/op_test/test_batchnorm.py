#pylint: disable=W0223, C0411, C0413, W0612, W0511, W0707
from __future__ import print_function
import logging
import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH']='OFF'
import copy
from itertools import product
import unittest

import torch
from torch import nn
import torch_mlu.core.mlu_model as ct

# W0223: class x(nn.Module) with not Implemented functions
# C0411, C0413: import order and sys path changes.
# W0612: unused params
# W0511: FIXMEs
# W0707: raise AssertionError

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase

# Supported params: Shape is NCDHW / NCHW / NCH
n_list = [4]
c_list = [32]
d_list = [4, 8]
hw_list = [14, 24]
affine_list = [True, False]
stats_list = [True, False]
momentum_list = [True, False]
test_backward_list = [True, False]
freeze_mode_list = [True, False]
dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
loop_val_infer = [affine_list, stats_list, momentum_list, dtype_list]
loop_val_train = loop_val_infer + [test_backward_list, freeze_mode_list]


# param is [shape, v1, v2, mom, (type, err)]
def inference_batchnormNd_impl(self, layer, shape):
    for param in product(*loop_val_infer):
        v1, v2, momentum, (data_type, err) = param
        tensor_in = torch.randn(shape).half().float()
        # bn in CPU
        bn = layer(shape[1], eps=1e-05, momentum=0.1, affine=v1, track_running_stats=v2)
        # set eval mode
        bn.eval()
        out_cpu = bn(tensor_in)
        # bn in MLU
        bn.to(ct.mlu_device()).to(data_type)
        out_mlu = bn(self.to_mlu_dtype(tensor_in, data_type))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

def inference_batchnormNd_impl_not_dense(self, layer, shape, num_features):
    for param in product(*loop_val_infer):
        v1, v2, momentum, (data_type, err) = param
        tensor_in = torch.randn(shape).half().float()
        # bn in CPU
        input_cpu = tensor_in[...,:int(shape[-1]/2)]
        bn = layer(num_features, eps=1e-05, momentum=0.1, affine=v1, track_running_stats=v2)
        # set eval mode
        bn.eval()
        out_cpu = bn(input_cpu)
        # bn in MLU
        bn.to(ct.mlu_device()).to(data_type)
        input_mlu = self.to_mlu_dtype(tensor_in, data_type)[...,:int(shape[-1]/2)]
        out_mlu = bn(input_mlu)

        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
        #self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())
        self.assertTrue(input_cpu.stride() == input_mlu.stride())
        self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())

def inference_batchnormNd_impl_channel_last(self, layer, shape):
    for param in product(*loop_val_infer):
        v1, v2, momentum, (data_type, err) = param
        tensor_in = torch.randn(shape).half().float()
        # bn in CPU
        input_cpu = tensor_in.to(memory_format=torch.channels_last)
        bn = layer(shape[1], eps=1e-05, momentum=0.1, affine=v1, track_running_stats=v2)
        # set eval mode
        bn.eval()
        out_cpu = bn(input_cpu)
        # bn in MLU
        bn.to(ct.mlu_device()).to(data_type)
        input_mlu = self.to_mlu_dtype(tensor_in, data_type).to(memory_format=torch.channels_last)
        out_mlu = bn(input_mlu)
        #self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())
        self.assertTrue(input_cpu.stride() == input_mlu.stride())
        self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

def training_batchnormNd_impl(self, layer, shape):
    for param in product(*loop_val_train):
        v1, v2, momentum, (data_type, err), test_backward, test_freeze = param
        weight_cpu = torch.randn(shape[1], dtype=torch.float)
        weight_mlu = copy.deepcopy(weight_cpu)
        class Net_cpu(nn.Module):
            def __init__(self):
                super(Net_cpu, self).__init__()
                self.features = layer(shape[1], affine=v1, track_running_stats=v2)
                if v1:
                    self.features.weight = nn.Parameter(weight_cpu)
                if momentum is False:
                    self.features.momentum = None
            def forward(self, x):
                output = self.features(x)
                return output
        class Net_mlu(nn.Module):
            def __init__(self):
                super(Net_mlu, self).__init__()
                self.features = layer(shape[1], affine=v1, track_running_stats=v2)
                if v1:
                    self.features.weight = nn.Parameter(weight_mlu)
                if momentum is False:
                    self.features.momentum = None
            def forward(self, x):
                output = self.features(x)
                return output

        model_cpu = Net_cpu()
        model_mlu = Net_mlu()
        if test_freeze:
            model_cpu.eval().float()
            model_mlu.eval().to(ct.mlu_device()).to(data_type)
        else:
            model_cpu.train().float()
            model_mlu.train().to(ct.mlu_device()).to(data_type)
        x_cpu = torch.randn(shape, dtype=torch.float, requires_grad=True)
        x_mlu = copy.deepcopy(x_cpu)
        out_cpu = model_cpu(x_cpu)
        out_mlu = model_mlu(self.to_mlu_dtype(x_mlu, data_type))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE = True)
        if test_backward:
            grad_cpu = torch.randn(shape, dtype=torch.float)
            grad_mlu = copy.deepcopy(grad_cpu)
            out_cpu.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x_cpu.grad)
            x_cpu.grad.zero_()
            out_mlu.backward(self.to_mlu_dtype(grad_mlu, data_type))
            x_grad_mlu = copy.deepcopy(x_mlu.grad)
            x_mlu.grad.zero_()
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu.cpu().float(), err, use_MSE=True)

def training_batchnormNd_impl_not_dense(self, layer, shape):
    for param in product(*loop_val_train):
        v1, v2, momentum, (data_type, err), test_backward, test_freeze = param
        weight_cpu = torch.randn(shape[1], dtype=torch.float)
        weight_mlu = copy.deepcopy(weight_cpu)
        class Net_cpu(nn.Module):
            def __init__(self):
                super(Net_cpu, self).__init__()
                self.features = layer(shape[1], affine=v1, track_running_stats=v2)
                if v1:
                    self.features.weight = nn.Parameter(weight_cpu)
                if momentum is False:
                    self.features.momentum = None
            def forward(self, x):
                output = self.features(x)
                return output
        class Net_mlu(nn.Module):
            def __init__(self):
                super(Net_mlu, self).__init__()
                self.features = layer(shape[1], affine=v1, track_running_stats=v2)
                if v1:
                    self.features.weight = nn.Parameter(weight_mlu)
                if momentum is False:
                    self.features.momentum = None
            def forward(self, x):
                output = self.features(x)
                return output

        model_cpu = Net_cpu()
        model_mlu = Net_mlu()
        if test_freeze:
            model_cpu.eval().float()
            model_mlu.eval().to(ct.mlu_device()).to(data_type)
        else:
            model_cpu.train().float()
            model_mlu.train().to(ct.mlu_device()).to(data_type)
        x_temp = torch.randn(shape, dtype=torch.float)
        x = torch.cat((x_temp, x_temp),-1)
        x_shape = x.shape
        x_cpu = x[...,:int(x_shape[-1]/2)].requires_grad_()
        x_mlu = self.to_mlu_dtype(x, data_type)[...,:int(x_shape[-1]/2)].requires_grad_()
        out_cpu = model_cpu(x_cpu)
        out_mlu = model_mlu(x_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE = True)
        #self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())
        self.assertTrue(x_cpu.stride() == x_mlu.stride())
        self.assertTrue(x_cpu.storage_offset() == x_mlu.storage_offset())
        if test_backward:
            grad = torch.randn(x_shape, dtype=torch.float)
            grad_cpu = grad[...,:int(x_shape[-1]/2)]
            grad_mlu = copy.deepcopy(grad)[...,:int(x_shape[-1]/2)]
            out_cpu.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x_cpu.grad)
            x_cpu.grad.zero_()
            out_mlu.backward(self.to_mlu_dtype(grad_mlu, data_type))
            x_grad_mlu = copy.deepcopy(x_mlu.grad)
            x_mlu.grad.zero_()
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu.cpu().float(), err, use_MSE=True)
            self.assertTrue(x_grad_cpu.stride() == x_grad_mlu.stride())
            self.assertTrue(x_grad_cpu.storage_offset() == x_grad_mlu.storage_offset())

def training_batchnormNd_impl_channel_last(self, layer, shape):
    for param in product(*loop_val_train):
        v1, v2, momentum, (data_type, err), test_backward, test_freeze = param
        weight_cpu = torch.randn(shape[1], dtype=torch.float)
        weight_mlu = copy.deepcopy(weight_cpu)
        class Net_cpu(nn.Module):
            def __init__(self):
                super(Net_cpu, self).__init__()
                self.features = layer(shape[1], affine=v1, track_running_stats=v2)
                if v1:
                    self.features.weight = nn.Parameter(weight_cpu)
                if momentum is False:
                    self.features.momentum = None
            def forward(self, x):
                output = self.features(x)
                return output
        class Net_mlu(nn.Module):
            def __init__(self):
                super(Net_mlu, self).__init__()
                self.features = layer(shape[1], affine=v1, track_running_stats=v2)
                if v1:
                    self.features.weight = nn.Parameter(weight_mlu)
                if momentum is False:
                    self.features.momentum = None
            def forward(self, x):
                output = self.features(x)
                return output

        model_cpu = Net_cpu()
        model_mlu = Net_mlu()
        if test_freeze:
            model_cpu.eval().float()
            model_mlu.eval().to(ct.mlu_device()).to(data_type)
        else:
            model_cpu.train().float()
            model_mlu.train().to(ct.mlu_device()).to(data_type)
        x = torch.randn(shape, dtype=torch.float)
        x_cpu = x.to(memory_format=torch.channels_last).requires_grad_()
        x_mlu = self.to_mlu_dtype(x, data_type)\
          .to(memory_format=torch.channels_last).requires_grad_()
        out_cpu = model_cpu(x_cpu)
        out_mlu = model_mlu(x_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE = True)
        #self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())
        self.assertTrue(x_cpu.stride() == x_mlu.stride())
        self.assertTrue(x_cpu.storage_offset() == x_mlu.storage_offset())
        if test_backward:
            grad = torch.randn(shape, dtype=torch.float)
            grad_cpu = grad.to(memory_format=torch.channels_last)
            grad_mlu = copy.deepcopy(grad).to(memory_format=torch.channels_last)
            out_cpu.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x_cpu.grad)
            x_cpu.grad.zero_()
            out_mlu.backward(self.to_mlu_dtype(grad_mlu, data_type))
            x_grad_mlu = copy.deepcopy(x_mlu.grad)
            x_mlu.grad.zero_()
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu.cpu().float(), err, use_MSE=True)
            self.assertTrue(x_grad_cpu.stride() == x_grad_mlu.stride())
            self.assertTrue(x_grad_cpu.storage_offset() == x_grad_mlu.storage_offset())

class TestBatchNormOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_inference_batchnorm1d(self):
        for n, c in product(n_list, c_list):
            # Test BN1d with 2d input
            shape = (n, c)
            inference_batchnormNd_impl(self, nn.BatchNorm1d, shape)
            # Test BN1d with 3d input
            for l in hw_list:
                shape = (n, c, l)
                inference_batchnormNd_impl(self, nn.BatchNorm1d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_inference_batchnorm1d_not_dense(self):
        for n, c in product(n_list, c_list):
            # Test BN1d with 2d input
            shape = (n, c*2)
            inference_batchnormNd_impl_not_dense(self, nn.BatchNorm1d, shape, c)
            # Test BN1d with 3d input
            for l in hw_list:
                shape = (n, c, l*2)
                inference_batchnormNd_impl_not_dense(self, nn.BatchNorm1d, shape, c)

    # @unittest.skip("not test")
    @testinfo()
    def test_inference_batchnorm2d(self):
        for n, c, hw in product(n_list, c_list, hw_list):
            shape = (n, c, hw, hw)
            inference_batchnormNd_impl(self, nn.BatchNorm2d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_inference_batchnorm2d_channel_last(self):
        for n, c, hw in product(n_list, c_list, hw_list):
            shape = (n, c, hw, hw)
            inference_batchnormNd_impl_channel_last(self, nn.BatchNorm2d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_inference_batchnorm2d_not_dense(self):
        for n, c, hw in product(n_list, c_list, hw_list):
            shape = (n, c, hw, hw)
            inference_batchnormNd_impl_not_dense(self, nn.BatchNorm2d, shape, c)

    # @unittest.skip("not test")
    @testinfo()
    def test_inference_batchnorm3d(self):
        for n, c, d, hw in product(n_list, c_list, d_list, hw_list):
            shape = (n, c, d, hw, hw)
            inference_batchnormNd_impl(self, nn.BatchNorm3d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_inference_batchnorm3d_not_dense(self):
        for n, c, d, hw in product(n_list, c_list, d_list, hw_list):
            shape = (n, c, d, hw, hw)
            inference_batchnormNd_impl_not_dense(self, nn.BatchNorm3d, shape, c)

    # @unittest.skip("not test")
    @testinfo()
    def test_training_batchnorm1d(self):
        for n, c in product(n_list, c_list):
            # Test BN1d with 2d input
            shape = (n, c)
            training_batchnormNd_impl(self, nn.BatchNorm1d, shape)
            # Test BN1d with 3d input
            for l in hw_list:
                shape = (n, c, l)
                training_batchnormNd_impl(self, nn.BatchNorm1d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_training_batchnorm1d_not_dense(self):
        for n, c in product(n_list, c_list):
            # Test BN1d with 2d input
            shape = (n, c)
            training_batchnormNd_impl_not_dense(self, nn.BatchNorm1d, shape,)
            # Test BN1d with 3d input
            for l in hw_list:
                shape = (n, c, l)
                training_batchnormNd_impl_not_dense(self, nn.BatchNorm1d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_training_batchnorm2d(self):
        for n, c, hw in product(n_list, c_list, hw_list):
            shape = (n, c, hw, hw)
            training_batchnormNd_impl(self, nn.BatchNorm2d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_training_batchnorm2d_channel_last(self):
        for n, c, hw in product(n_list, c_list, hw_list):
            shape = (n, c, hw, hw)
            training_batchnormNd_impl_channel_last(self, nn.BatchNorm2d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_training_batchnorm2d_not_dense(self):
        for n, c, hw in product(n_list, c_list, hw_list):
            shape = (n, c, hw, hw)
            training_batchnormNd_impl_not_dense(self, nn.BatchNorm2d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_training_batchnorm3d(self):
        for n, c, d, hw in product(n_list, c_list, d_list, hw_list):
            shape = (n, c, d, hw, hw)
            training_batchnormNd_impl(self, nn.BatchNorm3d, shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_batchnorm_exception(self):
        m = nn.BatchNorm2d(3)
        m.running_mean = m.running_mean.int()
        m.to('mlu')
        input = torch.randn((2, 3, 4, 4), dtype=torch.float).to('mlu')
        ref_msg = r"^running\_mean and running\_var need to have the same data types$"
        with self.assertRaisesRegex(RuntimeError,ref_msg):
            output = m(input)

if __name__ == '__main__':
    unittest.main()
