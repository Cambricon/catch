from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = "OFF" # pylint: disable=C0413
import copy
import unittest
import logging

import torch
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestLeakyReluOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_leaky_relu(self):
        for in_shape in [(11, 33), (8, 111, 131), (1, 1, 1, 1), (7, 3, 9, 16)]:
            # support fp32 and fp16, other dtype don't support, here we only test
            # fp32, so do below
            input_ = torch.randn(in_shape, dtype=torch.float)
            input_cpu = copy.deepcopy(input_)
            negative_slopes = [0.1, 0.01, 0.023, 0.5]
            for nega_val in negative_slopes:
                output_cpu = F.leaky_relu(input_cpu, negative_slope=nega_val)
                output_mlu = F.leaky_relu(input_.to("mlu"), negative_slope=nega_val)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
                self.assertTensorsEqual(input_cpu, input_, 0.003, use_MSE=True)

            input_mlu = input_.to("mlu")  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            for nega_val in negative_slopes:
                F.leaky_relu(input_cpu, inplace=True, negative_slope=nega_val)
                F.leaky_relu(input_mlu, inplace=True, negative_slope=nega_val)
                self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003, use_MSE=True)
                self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_leaky_relu_boundary_value(self):
        for number in [0, 0.0001, -0.0001, 99999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.leaky_relu(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.leaky_relu(x.to("mlu"))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, x, 0.003, use_MSE=True)

            input_cpu = copy.deepcopy(x)
            input_mlu = x.to("mlu")  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.leaky_relu(input_cpu, inplace=True)
            F.leaky_relu(input_mlu, inplace=True)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003, use_MSE=True)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_leaky_relu_backward(self):
        for shape in [(9, 17), (8, 224, 224), (1, 1, 1, 1), (8, 3, 16, 16)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = F.leaky_relu(x)
            out_mlu = F.leaky_relu(x.to("mlu"))
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu.backward(grad.to("mlu"))
            grad_mlu = x.grad
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_leaky_relu_backward_inplace(self):
        for shape in [(9, 17), (8, 224, 224), (1, 1, 1, 1), (8, 3, 16, 16)]:
            # In PyTorch 1.6, nega_val under zero is not supported for inplace LeakyReLU.
            for nega_val in [-0.5, 0.1, 0.2, 0.5, 0.99, 1.25, 10]:
                x_leaf = torch.randn(shape, dtype=torch.float, requires_grad=True)
                # A leaf tensor can't calculate grad after inplace operation
                x = x_leaf - 0.001
                x_mlu = x_leaf.to("mlu") - 0.001
                out_cpu = F.leaky_relu(x, inplace=True, negative_slope=nega_val)
                out_mlu = F.leaky_relu(x_mlu, inplace=True, negative_slope=nega_val)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                if nega_val >= 0:
                    out_cpu.backward(grad)
                    grad_cpu = copy.deepcopy(x_leaf.grad)
                    x_leaf.grad.zero_()
                    out_mlu.backward(grad.to("mlu"))
                    grad_mlu = x_leaf.grad
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    with self.assertRaises(RuntimeError):
                        out_mlu.backward(grad.to("mlu"))

    #@unittest.skip("not test")
    @testinfo()
    def test_leaky_relu_channels_last(self):
        for in_shape in [(3,8, 224, 224), (1, 1, 1, 1), (1, 3, 16, 16),
                         (1, 3, 16, 16)]:
            input_ = torch.randn(in_shape, dtype=torch.float).to(memory_format=torch.channels_last)
            output_cpu = F.leaky_relu(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.leaky_relu(self.to_mlu(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_, 0)

            input_cpu = copy.deepcopy(input_).to(memory_format = torch.channels_last)
            input_mlu = self.to_mlu(input_)  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.leaky_relu(input_cpu, inplace=True)
            F.leaky_relu(input_mlu, inplace=True)
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

        #@unittest.skip("not test")
    @testinfo()
    def test_leaky_relu_not_dense(self):
        for in_shape in [(2, 4), (8, 224, 224), (1, 1, 1, 8), (1, 3, 16, 16),
                         (1, 3, 16, 16, 10)]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            input_mlu = self.to_mlu(input_)[..., :2]
            input_cpu = input_[..., :2]
            output_cpu = F.leaky_relu(input_cpu)
            input_cpu_1 = copy.deepcopy(input_cpu)
            output_mlu = F.leaky_relu(input_mlu)
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu_1, input_cpu, 0)

            input_cpu = copy.deepcopy(input_)
            input_mlu = self.to_mlu(input_)[..., :2]  # test inplace operation
            input_cpu = input_cpu[..., :2]
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.leaky_relu(input_cpu, inplace=True)
            F.leaky_relu(input_mlu, inplace=True)
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

if __name__ == '__main__':
    unittest.main()
