from __future__ import print_function

import sys
import os
import copy
# import time
import unittest
import logging
# import numpy as np

import torch
# import torch.nn as nn
# from torch.nn import Parameter
# import torch.nn.functional as F
# import torch_mlu
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestTanhOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_tensor_scalar_contiguous(self):
        in_shape = [(10), (15, 19), (25, 19, 13), (13, 31, 16, 19), (14, 19, 21, 23, 21),
                    (16, 17, 18, 19, 20, 21)]
        for shape in in_shape:
            input_data = torch.randn(shape, dtype=torch.float)
            input_data_mlu = input_data.to(ct.mlu_device())

            output_cpu = torch.tanh(input_data)
            output_mlu = torch.tanh(input_data_mlu)

            # test scalar
            scalar_cpu = input_data.sum()
            scalar_mlu = scalar_cpu.to(ct.mlu_device())
            out_scalar_cpu = torch.tanh(scalar_cpu)
            out_scalar_mlu = torch.tanh(scalar_mlu)

            # test inplace operation
            input_mlu_ptr = input_data_mlu.data_ptr()
            input_data_mlu.tanh_()

            self.assertEqual(input_mlu_ptr, input_data_mlu.data_ptr())
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                out_scalar_cpu, out_scalar_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                output_cpu, input_data_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_tensor_scalar_channel_last(self):
        in_shape = [(13, 31, 16, 19), (14, 19, 21, 23, 21)]
        for shape in in_shape:
            input_data = torch.randn(shape, dtype=torch.float)
            input_data = self.convert_to_channel_last(input_data)
            input_data_mlu = input_data.to(ct.mlu_device())

            output_cpu = torch.tanh(input_data)
            output_mlu = torch.tanh(input_data_mlu)

            # test inplace operation
            input_mlu_ptr = input_data_mlu.data_ptr()
            input_data_mlu.tanh_()

            self.assertEqual(input_mlu_ptr, input_data_mlu.data_ptr())
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                output_cpu, input_data_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_tensor_scalar_not_dense(self):
        in_shape = [(15, 19 * 2), (25, 19, 13 * 2), (13, 31, 16, 19 * 2), (14, 19, 21, 23, 21 * 2),
                    (16, 17, 18, 19, 20, 21 * 2)]
        for shape in in_shape:
            input_data = torch.empty(0)
            if len(shape) == 2:
                input_data = torch.randn(shape, dtype=torch.float)[:, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                input_data = torch.randn(shape, dtype=torch.float)[:, :, :int(shape[-1] / 2)]
            elif len(shape) == 4:
                input_data = torch.randn(shape, dtype=torch.float)[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 5:
                input_data = torch.randn(shape, dtype=torch.float)[:, :, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 6:
                input_data = torch.randn(shape,\
                                          dtype=torch.float)[:, :, :, :, :, :int(shape[-1] / 2)]
            input_data = self.convert_to_channel_last(input_data)
            input_data_mlu = input_data.to(ct.mlu_device())

            output_cpu = torch.tanh(input_data)
            output_mlu = torch.tanh(input_data_mlu)

            # test inplace operation
            input_mlu_ptr = input_data_mlu.data_ptr()
            input_data_mlu.tanh_()

            self.assertEqual(input_mlu_ptr, input_data_mlu.data_ptr())
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                output_cpu, input_data_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_dtype(self):
        in_shape = [(10), (15, 19), (25, 19, 13), (13, 31, 16, 19), (14, 19, 21, 23, 21),
                    (16, 17, 18, 19, 20, 21)]
        # now cnnlTanh only support float and half
        type_list = [torch.float, torch.half]
        for shape in in_shape:
            for typeId in type_list:
                input_data = torch.randn(shape, dtype=torch.float)
                input_data_cpu = input_data.to(typeId)
                input_data_mlu = input_data_cpu.to(ct.mlu_device())

                output_cpu = torch.tanh(input_data)
                output_mlu = torch.tanh(input_data_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_backward(self):
        in_shape = [(50), (35, 46), (16, 27, 38), (128, 4, 128, 124), (14, 19, 11, 13, 21),
                    (6, 7, 8, 9, 10, 11), (16, 17, 18, 19, 20, 21)]
        type_list = [torch.float, torch.half]
        for shape in in_shape:
            for typeId in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(typeId)
                x_mlu = x.to(ct.mlu_device())

                # use float on cpu kernel
                out_cpu = x_0.tanh()
                out_mlu = x_mlu.tanh()

                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.to(ct.mlu_device())

                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)

                self.assertTensorsEqual(
                    out_grad_cpu,
                    out_grad_mlu.cpu().float() if typeId == torch.half else out_grad_mlu.cpu(),
                    0.003,
                    use_MSE=True)


if __name__ == '__main__':
    unittest.main()
