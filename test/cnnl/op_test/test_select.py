from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch
import copy                             # pylint: disable=C0411

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_select(self):
        in_shape = (2, 3, 24, 30)
        in_shape1 = (2, 3, 33)
        in_shape2 = (2, 24)
        input_dtypes = [torch.float, torch.half]
        for data_type in input_dtypes:
            input_t = torch.rand(in_shape, dtype=torch.float)
            input1 = torch.rand(in_shape1, dtype=torch.float)
            input2 = torch.rand(in_shape2, dtype=torch.float)
            output_cpu = input_t[:, 1]
            input_mlu = self.to_mlu_dtype(input_t, data_type)
            output_mlu = input_mlu[:, 1]
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)
            output_cpu1 = input1[:, 2]
            input1_mlu = self.to_mlu_dtype(input1, data_type)
            output_mlu1 = input1_mlu[:, 2]
            self.assertTensorsEqual(
                output_cpu1, output_mlu1.cpu().float(), 0.003, use_MSE=True)
            output_cpu1 = input2[1:, -1]
            input2_mlu = self.to_mlu_dtype(input2, data_type)
            output_mlu1 = input2_mlu[1:, -1]
            self.assertTensorsEqual(
                output_cpu1, output_mlu1.cpu().float(), 0.003, use_MSE=True)
            output_cpu = input_t[:, :, :, -2]
            output_mlu = input_mlu[:, :, :, -2]
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_select_channel_last(self):
        in_shape = (2, 3, 24, 30)
        input_dtypes = [torch.float, torch.half]
        for data_type in input_dtypes:
            input_t = torch.rand(in_shape).to(memory_format=torch.channels_last)
            output_cpu = input_t[:, 1]
            input_mlu = self.to_mlu_dtype(input_t, data_type)
            output_mlu = input_mlu[:, 1]
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)
            output_cpu = input_t[:, :, :, -2]
            output_mlu = input_mlu[:, :, :, -2]
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_select_not_dense(self):
        in_shape = (2, 3, 24, 30)
        input_dtypes = [torch.float, torch.half]
        for data_type in input_dtypes:
            input_t = torch.rand(in_shape)
            output_cpu = input_t[:,:,:,:19][:, :, :, 1]
            input_mlu = self.to_mlu_dtype(input_t, data_type)
            output_mlu = input_mlu[:,:,:,:19][:, :, :, 1]
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)
            output_cpu = input_t[:,:,:,:19][:, :, :, -2]
            output_mlu = input_mlu[:,:,:,:19][:, :, :, -2]
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_slice_backward(self):
        x = torch.randn((30, 2), requires_grad=True)
        x_mlu = self.to_device(x)
        z = x[12]
        z_mlu = x_mlu[12]
        grad = torch.randn(2)
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
    def test_select_exception(self):
        a = torch.tensor(4, dtype=torch.float).to('mlu')
        ref_msg = r"^select\(\) cannot be applied to a 0-dim tensor\.$"
        with self.assertRaisesRegex(IndexError, ref_msg):
            a.select(dim=1, index=2)

        a = torch.randn((3,4), dtype=torch.float).to('mlu')
        ref_msg = r"^select\(\): index 5 out of range for tensor of"
        ref_msg = ref_msg + r" size \[3, 4\] at dimension 1$"
        with self.assertRaisesRegex(IndexError, ref_msg):
            a.select(dim=1, index=5)

if __name__ == "__main__":
    unittest.main()
