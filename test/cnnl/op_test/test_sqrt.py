from __future__ import print_function

import sys
import os
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_sqrt(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (0), ()]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                out_cpu = torch.sqrt(x)
                out_mlu = torch.sqrt(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sqrt_channels_last(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4), (2, 3, 4, 20),
                      (254, 254, 112, 1, 1, 3), (0), ()]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                out_cpu = torch.sqrt(x)
                x_channel_last = self.convert_to_channel_last(x)
                out_mlu = torch.sqrt(self.to_mlu_dtype(x_channel_last, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float().contiguous(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sqrt_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (0), ()]
        data_types = [torch.float, torch.half]
        contiguous = [True, False]
        for shape in shape_list:
            for data_type in data_types:
                for con in contiguous:
                    x = torch.rand(shape, dtype=torch.float) + 0.01
                    if con is True:
                        x = self.get_not_contiguous_tensor(x)
                    out_cpu = torch.sqrt(x)
                    out_mlu = torch.sqrt(self.to_mlu_dtype(x, data_type))
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)


    # @unittest.skip("not test")
    @testinfo()
    def test_sqrt_(self):
        shape_list = [(2, 3, 4), (64, 3, 224)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                x_mlu = self.to_mlu_dtype(x, data_type)
                x_ptr = x_mlu.data_ptr()
                torch.sqrt_(x)
                torch.sqrt_(x_mlu)
                self.assertEqual(x_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True)
    # @unittest.skip("not test")
    @testinfo()
    def test_sqrt_inplace_channels_last(self):
        shape_list = [(2, 3, 4, 5), (64, 3, 224,30)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = (torch.rand(shape, dtype=torch.float) + 0.01).to(
                    memory_format = torch.channels_last)
                x_mlu = self.to_mlu_dtype(x, data_type)
                x_ptr = x_mlu.data_ptr()
                torch.sqrt_(x)
                torch.sqrt_(x_mlu)
                self.assertEqual(x_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sqrt_inplace_not_dense(self):
        shape_list = [(2, 3, 4, 5), (64, 3, 224, 30)]
        data_types = [torch.float, torch.half]
        contiguous = [True, False]
        for shape in shape_list:
            for data_type in data_types:
                for con in contiguous:
                    x = torch.rand(shape, dtype=torch.float) + 0.01
                    if con is True:
                        x = self.get_not_contiguous_tensor(x)
                    x_mlu = self.to_mlu_dtype(x, data_type)
                    x_ptr = x_mlu.data_ptr()
                    torch.sqrt_(x)
                    torch.sqrt_(x_mlu)
                    self.assertEqual(x_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(
                        x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True)

if __name__ == "__main__":
    unittest.main()
