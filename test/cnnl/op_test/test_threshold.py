from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
from itertools import product

import torch
from torch import nn
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestThresholdOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_threshold_contiguous(self):
        shape_list = [(4, 23, 13, 64), (7, 8, 8), (2, 3), (1,), ()]
        dtype_list = [torch.uint8, torch.int8, torch.int16, torch.float32,
                      torch.int32, torch.float64, torch.int64]
        list_list = [shape_list, dtype_list]
        for shape, dtype in product(*list_list):
            m = nn.Threshold(1, 2)
            # test forward
            if dtype.is_floating_point:
                input_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
            else:
                input_cpu = torch.randint(3, shape, dtype=dtype)
            input_mlu = copy.deepcopy(input_cpu)
            if dtype == torch.float16:
                output_cpu = m(input_cpu.float())
            else:
                output_cpu = m(input_cpu)
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
            if dtype.is_floating_point:
                # test backward
                grad = torch.randn(output_cpu.size())
                output_cpu.backward(grad)
                output_mlu.backward(self.to_device(grad))
                self.assertTensorsEqual(input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0)
            # test out
            if dtype.is_floating_point:
                output_mlu = self.to_device(torch.randn(shape, dtype=dtype))
            else:
                output_mlu = self.to_device(torch.randint(10, shape, dtype=dtype))
            with torch.no_grad():
                torch.threshold(self.to_device(input_mlu), 1, 2, out=output_mlu)
                self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)

        # test inplace
        m = nn.Threshold(1.1, 2.2, inplace=True)
        input_cpu = torch.randn(shape_list[1])
        input_mlu = input_cpu.to('mlu')
        data_ptr = input_mlu.data_ptr()
        m(input_cpu)
        m(input_mlu)
        self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
        self.assertEqual(data_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_threshold_channel_last(self):
        shape_list = [(4, 23, 13, 64)]
        dtype_list = [torch.uint8, torch.int8, torch.int16, torch.float32,
                      torch.int32, torch.float64, torch.int64]
        list_list = [shape_list, dtype_list]
        for shape, dtype in product(*list_list):
            m = nn.Threshold(1, 2)
            # test forward
            if dtype.is_floating_point:
                input_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
                with torch.no_grad():
                    input_cpu = self.convert_to_channel_last(input_cpu)
            else:
                input_cpu = torch.randint(3, shape, dtype=dtype)
            input_mlu = copy.deepcopy(input_cpu)
            if dtype == torch.float16:
                output_cpu = m(input_cpu.float())
            else:
                output_cpu = m(input_cpu)
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
            # test out
            if dtype.is_floating_point:
                output_mlu = self.to_device(torch.randn(shape, dtype=dtype))
            else:
                output_mlu = self.to_device(torch.randint(10, shape, dtype=dtype))
            with torch.no_grad():
                torch.threshold(self.to_device(input_mlu), 1, 2, out=output_mlu)
                self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)

        # test inplace
        m = nn.Threshold(1.1, 2.2, inplace=True)
        input_cpu = torch.randn(shape_list[0])
        input_cpu = self.convert_to_channel_last(input_cpu)
        input_mlu = input_cpu.to('mlu')
        data_ptr = input_mlu.data_ptr()
        m(input_cpu)
        m(input_mlu)
        self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
        self.assertEqual(data_ptr, input_mlu.data_ptr())

        #@unittest.skip("not test")
    @testinfo()
    def test_threshold_not_dense(self):
        shape_list = [(4, 23, 13, 64 * 2), (7, 8, 8 * 2), (2, 3 * 2)]
        dtype_list = [torch.uint8, torch.int8, torch.int16, torch.float32,
                      torch.int32, torch.float64, torch.int64]
        list_list = [shape_list, dtype_list]
        for shape, dtype in product(*list_list):
            m = nn.Threshold(1, 2)
            # test forward
            if dtype.is_floating_point:
                input_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
            else:
                input_cpu = torch.randint(3, shape, dtype=dtype)
            with torch.no_grad():
                if len(shape) == 2:
                    input_cpu = input_cpu[:, :int(shape[-1] / 2)]
                elif len(shape) == 3:
                    input_cpu = input_cpu[:, :, :int(shape[-1] / 2)]
                elif len(shape) == 4:
                    input_cpu = input_cpu[:, :, :, :int(shape[-1] / 2)]
            input_mlu = copy.deepcopy(input_cpu)
            if dtype == torch.float16:
                output_cpu = m(input_cpu.float())
            else:
                output_cpu = m(input_cpu)
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
            # test out
            if dtype.is_floating_point:
                output_mlu = self.to_device(torch.randn(shape, dtype=dtype))
            else:
                output_mlu = self.to_device(torch.randint(10, shape, dtype=dtype))
            with torch.no_grad():
                torch.threshold(self.to_device(input_mlu), 1, 2, out=output_mlu)
                self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)

        # test inplace
        m = nn.Threshold(1.1, 2.2, inplace=True)
        input_cpu = torch.randn(shape_list[1])
        input_cpu = self.convert_to_channel_last(input_cpu)
        input_mlu = input_cpu.to('mlu')
        data_ptr = input_mlu.data_ptr()
        m(input_cpu)
        m(input_mlu)
        self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
        self.assertEqual(data_ptr, input_mlu.data_ptr())

if __name__ == '__main__':
    unittest.main()
