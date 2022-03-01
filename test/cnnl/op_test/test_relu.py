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
import torch.nn.functional as F
# import torch_mlu
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestReluOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_relu(self):
        for in_shape in [(1), (2, 3), (8, 224, 224), (1, 1, 1, 1), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            output_cpu = F.relu(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.relu(self.to_mlu(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_, 0)

            input_cpu = copy.deepcopy(input_)
            input_mlu = self.to_mlu(input_)  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.relu(input_cpu, inplace=True)
            F.relu(input_mlu, inplace=True)
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_relu_channels_last(self):
        for in_shape in [(3,8, 224, 224), (1, 1, 1, 1), (1, 3, 16, 16),
                         (1, 3, 16, 16)]:
            input_ = torch.randn(in_shape, dtype=torch.float).to(memory_format=torch.channels_last)
            output_cpu = F.relu(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.relu(self.to_mlu(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_, 0)

            input_cpu = copy.deepcopy(input_).to(memory_format = torch.channels_last)
            input_mlu = self.to_mlu(input_)  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.relu(input_cpu, inplace=True)
            F.relu(input_mlu, inplace=True)
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

        #@unittest.skip("not test")
    @testinfo()
    def test_relu_not_dense(self):
        for in_shape in [(2, 4), (8, 224, 224), (1, 1, 1, 8), (1, 3, 16, 16),
                         (1, 3, 16, 16, 10)]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            input_mlu = self.to_mlu(input_)[..., :2]
            input_cpu = input_[..., :2]
            output_cpu = F.relu(input_cpu)
            input_cpu_1 = copy.deepcopy(input_cpu)
            output_mlu = F.relu(input_mlu)
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu_1, input_cpu, 0)

            input_cpu = copy.deepcopy(input_)
            input_mlu = self.to_mlu(input_)[..., :2]  # test inplace operation
            input_cpu = input_cpu[..., :2]
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.relu(input_cpu, inplace=True)
            F.relu(input_mlu, inplace=True)
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_relu_dtype(self):
        for in_shape in [(1), (2, 3), (8, 224, 224), (1, 1, 1, 1), (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            # cnnlActivation only support half/float, and cpu kernel don't support half
            dtypes = [torch.float, torch.half]
            for dtype in dtypes:
                input_ = torch.randn(in_shape).to(dtype)
                if dtype == torch.half:
                    input_ = input_.float()
                output_cpu = F.relu(input_)
                input_cpu = copy.deepcopy(input_)
                output_mlu = F.relu(self.to_mlu(input_))
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(input_cpu, input_, 0)

                input_mlu = self.to_mlu(input_)  # test inplace operation
                input_mlu_data_ptr = input_mlu.data_ptr()
                F.relu(input_cpu, inplace=True)
                F.relu(input_mlu, inplace=True)
                self.assertTensorsEqual(output_cpu, input_mlu.cpu().float(), 0)
                self.assertTensorsEqual(input_cpu, input_mlu.cpu().float(), 0)
                self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_relu_boundary_value(self):
        for number in [0, 0.0001, -0.0001, 999999999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.relu(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.relu(self.to_mlu(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, x, 0)

            input_cpu = copy.deepcopy(x)
            input_mlu = self.to_mlu(x)  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.relu(input_cpu, inplace=True)
            F.relu(input_mlu, inplace=True)
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_relu_backward(self):
        for shape in [(1), (2, 3), (8, 224, 224), (1, 1, 1, 1), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = x.relu()
            out_mlu = self.to_mlu(x).relu()
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x_cpu = copy.deepcopy(x)
            x.grad.zero_()
            out_mlu.backward(self.to_mlu(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)
            self.assertTensorsEqual(x, x_cpu, 0)

            x.grad.zero_()  # test inplace operation
            x_mlu = self.to_mlu(x)
            x_mlu.relu_()
            x_mlu.backward(self.to_mlu(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)
            self.assertTensorsEqual(x, x_cpu, 0)


if __name__ == '__main__':
    unittest.main()
