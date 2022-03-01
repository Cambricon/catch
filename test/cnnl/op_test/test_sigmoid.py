from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestSigmoidOp(TestCase):

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid(self):
        # mlu device support torch.half, while cpu not
        type_list = [
            torch.float
        ]
        for Type in type_list:
            # test_dim_0
            x_0 = torch.tensor(8.0, dtype=Type)

            out_cpu = torch.sigmoid(x_0)
            out_mlu = torch.sigmoid(copy.deepcopy(x_0).to(ct.mlu_device()))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003)

            for in_shape in [(1), (2,3),
                             (8, 224, 224),
                             (1,1,1,1),
                             (1, 3, 16, 16),
                             (1, 3, 16, 16, 3)]:
                input_mlu = torch.randn(in_shape, dtype=Type)
                input_cpu = copy.deepcopy(input_mlu)
                input_mlu_raw = copy.deepcopy(input_mlu)

                output_cpu = torch.sigmoid(input_cpu)
                output_mlu = torch.sigmoid(input_mlu.to(ct.mlu_device()))

                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
                self.assertTensorsEqual(input_mlu_raw, input_mlu, 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_channel_last(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_mlu = torch.randn(in_shape).to(memory_format=torch.channels_last)
            input_cpu = copy.deepcopy(input_mlu)
            input_mlu_raw = copy.deepcopy(input_mlu)

            output_cpu = torch.sigmoid(input_cpu)
            output_mlu = torch.sigmoid(input_mlu.to(ct.mlu_device()))

            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_mlu_raw, input_mlu, 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_not_dense(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_mlu = torch.randn(in_shape)
            input_cpu = copy.deepcopy(input_mlu)
            input_mlu_raw = copy.deepcopy(input_mlu)
            output_cpu = torch.sigmoid(input_cpu[:,:,:,:2])
            output_mlu = torch.sigmoid(input_mlu.to(ct.mlu_device())[:,:,:,:2])

            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_mlu_raw, input_mlu, 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_inplace(self):
        # mlu device support torch.half, while cpu not
        type_list = [
            torch.float
        ]
        for Type in type_list:
            # test_dim_0
            x_0 = torch.tensor(8.0, dtype=Type)

            out_cpu = x_0
            out_mlu = copy.deepcopy(x_0).to(ct.mlu_device())
            out_cpu.sigmoid_()
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu.sigmoid_()
            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003)

            for in_shape in [(1), (2, 3),
                             (8, 224, 224),
                             (1, 1, 1, 1),
                             (1, 3, 16, 16),
                             (1, 3, 16, 16, 3)]:
                input_cpu = torch.randn(in_shape, dtype=Type)
                input_mlu = copy.deepcopy(input_cpu).to(ct.mlu_device())

                input_cpu.sigmoid_()
                input_mlu_ptr = input_mlu.data_ptr()
                input_mlu.sigmoid_()

                self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
                self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_inplace_channel_last(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_cpu = torch.randn(in_shape).to(memory_format=torch.channels_last)
            input_mlu = copy.deepcopy(input_cpu).to(ct.mlu_device())

            input_cpu.sigmoid_()
            input_mlu_ptr = input_mlu.data_ptr()
            input_mlu.sigmoid_()

            self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_inplace_not_dense(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_cpu = torch.randn(in_shape)
            input_mlu = copy.deepcopy(input_cpu).to(ct.mlu_device())

            input_cpu[:,:,:,:2].sigmoid_()
            input_mlu_ptr = input_mlu.data_ptr()
            input_mlu[:,:,:,:2].sigmoid_()

            self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_backward(self):
        for in_shape in [(1), (2, 3),
                         (8, 224, 224),
                         (1, 1, 1, 1),
                         (1, 3, 16, 16),
                         (1, 3, 16, 16, 3)]:
            x = torch.randn(in_shape, dtype=torch.float, requires_grad=True)
            x_mlu = x.to(ct.mlu_device())

            # use float on cpu kernel
            out_cpu = x.sigmoid()
            out_mlu = x_mlu.sigmoid()

            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.to(ct.mlu_device())

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()

            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(
                out_grad_cpu,
                out_grad_mlu.cpu().float(),
                0.003
            )

if __name__ == '__main__':
    unittest.main()
