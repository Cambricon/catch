from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)

class TestPoolbpOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_avgpooling_backward(self):
        shape_list = [(8, 16, 7, 7), (16, 6, 8, 16), (4, 23, 13, 64)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [2]
        stride_v = [3, None]
        padding_v = [0]
        ceil_mode_v = [False, True]
        include_pad_v = [False]

        loop_var = [
            shape_list, memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v,
            include_pad_v
        ]
        for in_shape, memory_format, kernel, stride, padding, ceil_mode, include_pad in product(
                *loop_var):
            input_t = torch.randn(in_shape,
                                  dtype=torch.float).to(memory_format = memory_format)
            avg_pool = nn.AvgPool2d(kernel,
                                    stride=stride,
                                    padding=padding,
                                    ceil_mode=ceil_mode,
                                    count_include_pad=include_pad)
            input_t.requires_grad=True
            input_mlu = copy.deepcopy(input_t)
            output_cpu = avg_pool(input_t)
            grad = torch.randn(output_cpu.shape, dtype=torch.float).to(memory_format = memory_format)
            output_cpu.backward(grad, retain_graph = True)

            output_mlu = avg_pool(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu(),
                                    3e-3,
                                    use_MSE=True)

            output_mlu.backward(self.to_device(grad), retain_graph = True)
            self.assertTensorsEqual(input_t.grad.float(), input_mlu.grad.float(), 0.003, use_RAE=True)

            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1).to(memory_format = memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_device(grad)[..., :-1])
            self.assertTensorsEqual(input_t.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_RAE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpooling_backward(self):
        in_shape = (1, 1, 8, 8)
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3]
        stride_v = [2]
        padding_v = [1]
        ceil_mode_v = [False]
        return_indices_v = [False]

        loop_var = [
            memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v, return_indices_v
        ]
        for memory_format, kernel, stride, padding, ceil_mode, return_indices in product(
                *loop_var):
            input_t = torch.randn(in_shape,
                                  dtype=torch.float).to(memory_format = memory_format)
            input_t.requires_grad=True
            input_mlu = copy.deepcopy(input_t)
            output_cpu = F.max_pool2d(input_t,
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph = True)
            grad_cpu = input_t.grad
            
            output_mlu = F.max_pool2d(self.to_device(input_mlu),
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu(),
                                    3e-3,
                                    use_MSE=True)
            output_mlu.backward(self.to_device(grad), retain_graph = True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # test not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1).to(memory_format = memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_device(grad)[..., :-1])
            self.assertTensorsEqual(input_t.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_RAE=True)


    # @unittest.skip("not test")
    @testinfo()
    def test_avgpooling3d_backward(self):
        shape_list = [(12, 2048, 1, 7, 7),
                      (12, 192, 8, 28, 28)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(1, 7, 7), (1, 3, 3)]
        stride_v = [(1, 1, 1), (1, 1, 1)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        include_pad_v = [True, True]

        loop_var = [
            shape_list, memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v,
            include_pad_v
        ]
        for in_shape, memory_format, kernel, stride, padding, ceil_mode, include_pad in zip(
                *loop_var):
            input_t = torch.randn(in_shape, dtype=torch.float).to(memory_format = memory_format)
            input_t.requires_grad=True
            input_mlu = copy.deepcopy(input_t)
            # test nn module
            avg_pool = nn.AvgPool3d(kernel,
                                    stride=stride,
                                    padding=padding,
                                    ceil_mode=ceil_mode,
                                    count_include_pad=include_pad)
            output_cpu = avg_pool(input_t)
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph = True)
            grad_cpu = input_t.grad

            output_mlu = avg_pool(self.to_mlu(input_mlu))
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)
            output_mlu.backward(self.to_device(grad), retain_graph = True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)
            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3], o_shape[4] + 1).to(memory_format = memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_device(grad)[..., :-1])
            self.assertTensorsEqual(input_t.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_RAE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpooling3d_backward(self):
        shape_list = [(12, 2048, 2, 7, 7),
                      (12, 128, 8, 112, 112)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(2, 1, 1), (2, 3, 3)]
        stride_v = [(1, 1, 1), (2, 2, 2)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        return_indices_v = [False, False]

        loop_var = [
            shape_list, memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v, return_indices_v
        ]
        for in_shape, memory_format, kernel, stride, padding, ceil_mode, return_indices in zip(
                *loop_var):
            input_t = torch.randn(in_shape, dtype=torch.float).to(memory_format = memory_format)
            input_t.requires_grad=True
            input_mlu = copy.deepcopy(input_t)
            # test nn module
            max_pool = nn.MaxPool3d(kernel,
                                    stride=stride,
                                    padding=padding,
                                    dilation=1,
                                    ceil_mode=ceil_mode,
                                    return_indices=return_indices)
            output_cpu = max_pool(input_t)
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph = True)
            grad_cpu = input_t.grad

            output_mlu = max_pool(self.to_mlu(input_mlu))
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)
            output_mlu.backward(self.to_device(grad), retain_graph = True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3], o_shape[4] + 1).to(memory_format = memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_device(grad)[..., :-1])
            self.assertTensorsEqual(input_t.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_RAE=True)


if __name__ == '__main__':
    unittest.main()
