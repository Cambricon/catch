from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
import torch.nn.functional as F
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestSoftplusOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_softplus(self):
        for in_shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            output_cpu = F.softplus(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.softplus(self.to_device(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_channels_last(self):
        for in_shape in [(2, 3, 24, 30),(1, 1, 1, 30)]:
            input_ = torch.randn(in_shape, dtype=torch.float).to(
                memory_format = torch.channels_last)
            output_cpu = F.softplus(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.softplus(self.to_device(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_not_dense(self):
        for in_shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            for con in [True, False]:
                input_ = torch.randn(in_shape, dtype=torch.float)
                if con is True:
                    input_ = self.get_not_contiguous_tensor(input_)
                output_cpu = F.softplus(input_)
                input_cpu = copy.deepcopy(input_)
                output_mlu = F.softplus(self.to_device(input_))
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
                self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_dtype(self):
        for in_shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            dtypes = [torch.float, torch.double]
            for dtype in dtypes:
                input_ = torch.randn(in_shape).to(dtype)
                output_cpu = F.softplus(input_)
                input_cpu = copy.deepcopy(input_)
                output_mlu = F.softplus(self.to_device(input_))
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
                self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_boundary_value(self):
        for number in [0, 0.0001, -0.0001, 999999999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.softplus(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.softplus(self.to_device(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, x, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_backward(self):
        for shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = F.softplus(x,beta=1, threshold=20)
            out_mlu = F.softplus(self.to_device(x),beta=1, threshold=20)
            out_mlu_ptr = out_mlu.data_ptr()
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x_cpu = copy.deepcopy(x)
            x.grad.zero_()
            out_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(x, x_cpu, 0)



if __name__ == '__main__':
    unittest.main()
