from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import unittest
import logging

import torch
import torch.autograd
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class LogSoftMaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, result=None):
        if ct.is_mlu_tensor(x):
            result = torch.log_softmax(x,dim)
        else:
            if result is None:
                logging.error("logsoftmaxbackward requires result!!")
        ctx.save_for_backward(x, result)
        ctx.dim = dim
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x,result = ctx.saved_tensors
        dim = ctx.dim
        grad = torch._log_softmax_backward_data(grad_output, result, dim, x)
        return grad

class TestOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_softmax(self):
        shapes = [(2, 3, 4, 5, 7, 8, 9, 11),(2, 3, 4, 5), (2, 3, 4), (2, 3), (2, )]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                for dim in range(len(shape)):
                    y_mlu = torch.nn.functional.softmax(x_mlu, dim)
                    y_cpu = torch.nn.functional.softmax(x_cpu, dim)
                    self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

        for data_type, err in dtype_list:
            x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
            x_mlu = self.to_mlu_dtype(x_cpu, data_type)
            dims = [-3, -2, -1, 0, 1, 2, 3]
            for i in range(len(dims)):  # pylint: disable=C0200
                y_mlu = torch.nn.functional.softmax(x_mlu, dims[i])
                y_cpu = torch.nn.functional.softmax(x_cpu, dims[i])
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax_channels_last(self):
        shapes = [(2, 3, 4, 5), (2, 3, 24, 30), (1, 1, 1, 30)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float).to(
                    memory_format = torch.channels_last)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                for dim in range(len(shape)):
                    y_mlu = torch.nn.functional.softmax(x_mlu, dim)
                    y_cpu = torch.nn.functional.softmax(x_cpu, dim)
                    self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

        for data_type, err in dtype_list:
            x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
            x_mlu = self.to_mlu_dtype(x_cpu, data_type)
            dims = [-3, -2, -1, 0, 1, 2, 3]
            for i in range(len(dims)):  # pylint: disable=C0200
                y_mlu = torch.nn.functional.softmax(x_mlu, dims[i])
                y_cpu = torch.nn.functional.softmax(x_cpu, dims[i])
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_softmax_not_dense(self):
        shapes = [(2, 3, 4, 5), (2, 3, 4), (2, 3)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                for dim in range(len(shape)):
                    y_mlu = torch.nn.functional.softmax(x_mlu[:,:2], dim)
                    y_cpu = torch.nn.functional.softmax(x_cpu[:,:2], dim)
                    self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_softmax_backward(self):
        shapes = [(2, 3, 4, 5), (2, 3, 4), (2, 3), (2, ), ()]
        for shape in shapes:
            for dim in range(max(len(shape), 1)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = x.softmax(dim)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()
                out_mlu = self.to_mlu(x).softmax(dim)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu, 0.003, use_MSE=True)
        # test empty tensor
        x = torch.randn([], dtype=torch.float, requires_grad=True)
        out_cpu = x.softmax(0)
        grad = torch.randn(out_cpu.shape, dtype=torch.float)
        out_cpu.backward(grad)
        grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu = self.to_mlu(x).softmax(0)
        out_mlu.backward(self.to_mlu(grad))
        grad_mlu = copy.deepcopy(x.grad)
        self.assertTensorsEqual(
            grad_cpu, grad_mlu, 0.003, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_logsoftmax(self):
        shapes = [(64, 1000), (16, 5, 7), (2, 3, 4, 5), (2, ), ()]
        log_softmax_cpu = LogSoftMaxFunc()
        log_softmax_mlu = LogSoftMaxFunc()
        for shape in shapes:
            for dim in range(max(len(shape), 1)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_mlu = log_softmax_mlu.apply(self.to_mlu(x), dim)
                out_cpu = log_softmax_cpu.apply(x, dim, out_mlu.cpu())
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                grad_cpu = out_cpu.grad_fn.apply(grad)
                grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
        dims = [-3, -2, -1, 0, 1, 2, 3]
        for i in range(len(dims)):  # pylint: disable=C0200,W0612
            x = torch.randn(2, 3, 4, 5, dtype=torch.float, requires_grad=True)
            out_mlu = log_softmax_mlu.apply(self.to_mlu(x), dim)
            out_cpu = log_softmax_cpu.apply(x, dim, out_mlu.cpu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_cpu = out_cpu.grad_fn.apply(grad)
            grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_channels_last(self):
        shapes = [(2, 3, 4, 5), (2, 3, 24, 30), (1, 1, 1, 30)]
        log_softmax_cpu = LogSoftMaxFunc()
        log_softmax_mlu = LogSoftMaxFunc()
        for shape in shapes:
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True).to(
                    memory_format = torch.channels_last)
                out_mlu = log_softmax_mlu.apply(self.to_mlu(x), dim)
                out_cpu = log_softmax_cpu.apply(x, dim, out_mlu.cpu())
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                grad_cpu = out_cpu.grad_fn.apply(grad)
                grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_not_dense(self):
        shapes = [(64, 1000), (16, 5, 7), (2, 3, 4, 5)]
        log_softmax_cpu = LogSoftMaxFunc()
        log_softmax_mlu = LogSoftMaxFunc()
        for shape in shapes:
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_mlu = log_softmax_mlu.apply(self.to_mlu(x)[:,:2], dim)
                out_cpu = log_softmax_cpu.apply(x[:,:2], dim, out_mlu.cpu())
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                grad_cpu = out_cpu.grad_fn.apply(grad)
                grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_exception(self):
        a = torch.randn(3, dtype=torch.float).to('mlu')
        ref_msg = r"^conversion is supported for Half type only$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch._log_softmax(a, dim=0, half_to_float=True)

if __name__ == "__main__":
    unittest.main()
