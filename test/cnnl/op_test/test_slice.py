from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=all
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_slice(self):
        in_shape = (2, 3, 24, 30)
        in_shape1 = (2, 3, 33)
        in_shape2 = (2, 24)
        input_dtypes = [torch.float, torch.half]
        channel_first = [False, True]
        for data_type in input_dtypes:
            for channel in channel_first:
                input_t = torch.rand(in_shape, dtype=torch.float)
                input1 = torch.rand(in_shape1, dtype=torch.float)
                input2 = torch.rand(in_shape2, dtype=torch.float)
                output_cpu = input_t[:, 1:, 2:-1:3, 10:20]
                if channel is False:
                    input_t = self.convert_to_channel_last(input_t)
                input_mlu = self.to_mlu_dtype(input_t, data_type)
                output_mlu = input_mlu[:, 1:, 2:-1:3, 10:20]
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)
                output_cpu1 = input1[:, 1:, :]
                if channel is False:
                    input1 = self.convert_to_channel_last(input1)
                input1_mlu = self.to_mlu_dtype(input1, data_type)
                output_mlu1 = input1_mlu[:, 1:, :]
                self.assertTensorsEqual(
                    output_cpu1, output_mlu1.cpu().float(), 0.003, use_MSE=True)
                output_cpu1 = input2[1:, 10:]
                if channel is False:
                    input2 = self.convert_to_channel_last(input2)
                input2_mlu = self.to_mlu_dtype(input2, data_type)
                output_mlu1 = input2_mlu[1:, 10:]
                self.assertTensorsEqual(
                    output_cpu1, output_mlu1.cpu().float(), 0.003, use_MSE=True)
                output_cpu = input_t[:, :, :, -2:]
                output_mlu = input_mlu[:, :, :, -2:]
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_slice_backward(self):
        x = torch.randn((30, 2), requires_grad=True)
        x_mlu = self.to_device(x)
        z = x[1:12]
        z_mlu = x_mlu[1:12]
        grad = torch.randn(11, 2)
        grad_mlu = self.to_device(grad)
        z.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        x.grad.zero_()
        z_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(
              z, z_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(
              out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

        x = torch.randn((5, 2), requires_grad=True)
        x_mlu = self.to_device(x)
        z = x[1:]
        z_mlu = x_mlu[1:]
        grad = torch.randn(4, 2)
        grad_mlu = self.to_device(grad)
        z.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        x.grad.zero_()
        z_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(
              z, z_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(
              out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_slice_exception(self):
        a = torch.tensor(5, dtype=torch.float).to('mlu')
        ref_msg = r"^dimension specified as 0 but tensor has no dimensions$"
        with self.assertRaisesRegex(IndexError, ref_msg):
            b = a[0:1:1]

        a = torch.randn((2,3,4), dtype=torch.float).to('mlu')
        ref_msg = r"^step must be greater than zero$"
        with self.assertRaisesRegex(ValueError, ref_msg):
            b = a[0:1:-1, :]

if __name__ == "__main__":
    unittest.main()
