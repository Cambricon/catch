from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct       # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestSqueezeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for in_shape in [(2, 1, 2, 1, 2), (2, 3, 4)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape, dtype=torch.float)
                output_cpu = torch.squeeze(input1)
                output_mlu = torch.squeeze(self.to_mlu_dtype(input1, data_type))
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_channel_last(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for in_shape in [(2, 1, 2, 1), (2, 3, 4, 5)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape).to(memory_format=torch.channels_last)
                output_cpu = torch.squeeze(input1)
                output_mlu = torch.squeeze(self.to_mlu_dtype(input1, data_type))
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_not_dense(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for in_shape in [(4, 5, 1, 3, 4), (2, 3, 4)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape, dtype=torch.float)
                output_cpu = torch.squeeze(input1[::2])
                output_mlu = torch.squeeze(self.to_mlu_dtype(input1, data_type)[::2])
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_inplace(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for in_shape in [(2, 1, 2, 1, 2)]:
            for dim in [1, 3]:
                for data_type, err in dtype_list:
                    input_t = torch.randn(in_shape, dtype=torch.float)
                    input_mlu = copy.deepcopy(input_t).to('mlu').to(data_type)
                    input_t.squeeze_(dim)
                    input_mlu.squeeze_(dim)
                    self.assertTensorsEqual(input_t, input_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_inplace_channel_last(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for in_shape in [(4, 5, 3, 4)]:
            for dim in [1, 3]:
                for data_type, err in dtype_list:
                    input_t = torch.randn(in_shape).to(memory_format=torch.channels_last)
                    input_mlu = copy.deepcopy(input_t).to('mlu').to(data_type)
                    input_t.squeeze_(dim)
                    input_mlu.squeeze_(dim)
                    self.assertTensorsEqual(input_t, input_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_inplace_not_dense(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for in_shape in [(4, 5, 1, 3, 4)]:
            for dim in [1, 3]:
                for data_type, err in dtype_list:
                    input_t = torch.randn(in_shape, dtype=torch.float)
                    input_mlu = copy.deepcopy(input_t).to('mlu').to(data_type)
                    input_t[::2].squeeze_(dim)
                    input_mlu[::2].squeeze_(dim)
                    self.assertTensorsEqual(input_t, input_mlu.cpu().float(), err)

if __name__ == '__main__':
    unittest.main()
