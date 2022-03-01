from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct   # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_repeat(self):
        for in_shape in [(1, 2, 2), (2, 3, 4, 5), (10, 10, 10),
                         (4, 5, 3), (2, 2), (3,), (5, 6, 10, 24, 24)]:
            for repeat_size in [(2, 3, 4), (2, 3, 4, 5), (2, 2, 2, 2, 2)]:
                if len(repeat_size) < len(in_shape):
                    continue
                input1 = torch.randn(in_shape, dtype=torch.float)
                output_cpu = input1.repeat(repeat_size)
                output_mlu = self.to_mlu(input1).repeat(repeat_size)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-4, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_channels_last(self):
        for in_shape in [(2, 3, 4, 5), (1, 2, 2), (2, 2), (3,), (5, 6, 10, 24, 24),()]:
            for repeat_size in [(2, 3, 4), (2, 3, 2, 4,), (2, 2, 2, 2, 2)]:
                if len(repeat_size) < len(in_shape):
                    continue
                input1 = torch.randn(in_shape, dtype=torch.float)
                channels_last_input1 = self.convert_to_channel_last(input1)
                output_cpu = channels_last_input1.repeat(repeat_size)
                output_mlu = self.to_mlu(channels_last_input1).repeat(repeat_size)
                output_mlu_channels_first = output_mlu.cpu().float().contiguous()
                self.assertTensorsEqual(
                    output_cpu, output_mlu_channels_first, 3e-4, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_not_dense(self):
        for in_shape in [(10, 10, 10), (4, 5, 3)]:
            for repeat_size in [(2, 3, 4), (2, 3, 4, 5), (2, 2, 2, 2, 2)]:
                if len(repeat_size) < len(in_shape):
                    continue
                input1 = torch.randn(in_shape, dtype=torch.float)
                output_cpu = input1[:, :, :2].repeat(repeat_size)
                output_mlu = self.to_mlu(input1)[:, :, :2].repeat(repeat_size)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-4, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_backward(self):
        N_lst = [4, 16]
        HW_lst = [16, 64]
        C_lst = [2, 8]
        sizes = [3, 5, 3, 5]
        for N in N_lst:
            for HW in HW_lst:
                for C in C_lst:
                    x = torch.randn(
                        N, C, HW, HW, dtype=torch.float, requires_grad=True)
                    out_cpu = x.repeat(sizes)
                    grad = torch.randn(out_cpu.shape, dtype=torch.float)
                    out_cpu.backward(grad)
                    grad_cpu = copy.deepcopy(x.grad)
                    x.grad.zero_()
                    out_mlu = self.to_mlu(x).repeat(sizes)
                    out_mlu.backward(self.to_mlu(grad))
                    grad_mlu = copy.deepcopy(x.grad)
                    self.assertTensorsEqual(
                        grad_cpu, grad_mlu, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_exception(self):
        a = torch.randn((1, 2, 2)).to('mlu')
        ref_msg = r"Number of dimensions of repeat dims can not be smaller than number"
        ref_msg = ref_msg + " of dimensions of tensor"
        repeat_size = (1, 2)
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.repeat(repeat_size)

if __name__ == "__main__":
    unittest.main()
