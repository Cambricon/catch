from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_resize_inplace(self):
        dtypes = [torch.float, torch.int64, torch.long, torch.half]
        for in_shape, out_shape in [((1, 10, 1, 1), (2, 2)),
                                    ((20, 20), (400, 1)),
                                    ((1, 4, 3, 2), (2, 3, 4))]:
            for dtype in dtypes:
                x_cpu = torch.randn(in_shape).to(dtype)
                x_mlu = self.to_device(x_cpu)
                y_cpu = x_cpu.resize_(out_shape)
                y_mlu = x_mlu.resize_(out_shape)
                self.assertTrue(y_cpu.size() == y_mlu.size())
                self.assertTrue(y_cpu.stride() == y_mlu.stride())
                self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
                self.assertTrue(x_cpu.size() == x_mlu.cpu().size())
                self.assertTrue(x_cpu.stride() == x_mlu.cpu().stride())
                self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_resize_inplace_channel_last(self):
        dtypes = [torch.float, torch.int64, torch.long, torch.half]
        for in_shape, out_shape in [((1, 10, 1, 1), (2, 2)),
                                    ((1, 4, 3, 2), (2, 3, 4))]:
            for dtype in dtypes:
                x_cpu = torch.randn(in_shape).to(dtype).to(memory_format=torch.channels_last)
                x_mlu = self.to_device(x_cpu)
                y_cpu = x_cpu.resize_(out_shape)
                y_mlu = x_mlu.resize_(out_shape)
                self.assertTrue(y_cpu.size() == y_mlu.size())
                self.assertTrue(y_cpu.stride() == y_mlu.stride())
                self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
                self.assertTrue(x_cpu.size() == x_mlu.cpu().size())
                self.assertTrue(x_cpu.stride() == x_mlu.cpu().stride())
                self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_resize_inplace_not_dense(self):
        dtypes = [torch.float, torch.int64, torch.long, torch.half]
        for in_shape, out_shape in [((1, 4, 3, 2), (2, 3, 4))]:
            for dtype in dtypes:
                x_cpu = torch.randn(in_shape).to(dtype)
                x_cpu = self.get_not_contiguous_tensor(x_cpu)
                x_mlu = self.to_device(x_cpu)
                y_cpu = x_cpu.resize_(out_shape)
                y_mlu = x_mlu.resize_(out_shape)
                self.assertTrue(y_cpu.size() == y_mlu.size())
                self.assertTrue(y_cpu.stride() == y_mlu.stride())
                self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
                self.assertTrue(x_cpu.size() == x_mlu.cpu().size())
                self.assertTrue(x_cpu.stride() == x_mlu.cpu().stride())
                self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

if __name__ == "__main__":
    unittest.main()
