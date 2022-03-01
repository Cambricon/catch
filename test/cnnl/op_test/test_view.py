from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_view(self):
        for in_shape, out_shape in [((1, 1000, 1, 1), (1, -1)),
                                    ((1, 3, 200, 200), (1, -1, 1, 200, 200)),
                                    ((1, ), (1, )),
                                    ((1, 58, 2, 28, 28), (2, -1, 4)),
                                    ((45, 54, 454), (45, 54, 454))]:
            x_cpu = torch.randint(0, 100, (in_shape))
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu.view(out_shape)
            y_mlu = x_mlu.view(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_channels_last_and_not_dense(self):
        for in_shape, out_shape in [((1, 1000, 1, 1), (1, -1)),
                                    ((2, 3, 4, 5), (2, -1, 1, 4, 5))]:
            x_cpu = torch.randn(in_shape).to(memory_format = torch.channels_last)
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu[:,:2].view(out_shape)
            y_mlu = x_mlu[:,:2].view(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_channels_last_unsafe(self):
        for in_shape, out_shape in [((64, 3, 24, 24), (1, -1)),
                                    ((2, 3, 4, 5), (-1, 5)),
                                    ((13, 77, 23, 153), (23, 1001, 153))]:
            x_cpu = torch.randn(in_shape)
            x_cl = x_cpu.to(memory_format = torch.channels_last)
            x_mlu = self.to_device(x_cl)
            y_cpu = x_cpu.view(out_shape)
            y_mlu = x_mlu.view(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_dtype(self):
        # half -> int16
        x = torch.randn(5, 5, dtype=torch.half)
        x_mlu = x.to('mlu')
        out_mlu = torch.ops.torch_mlu.view(x_mlu, torch.int16)
        base_cpu = x.numpy().view(np.int16)
        self.assertEqual(out_mlu.cpu(), base_cpu, 0)

        # float -> int32
        x = torch.randn(5, 5, dtype=torch.float)
        x_mlu = x.to('mlu')
        out_mlu = torch.ops.torch_mlu.view(x_mlu, torch.int32)
        base_cpu = x.numpy().view(np.int32)
        self.assertEqual(out_mlu.cpu(), base_cpu, 0)



if __name__ == "__main__":
    unittest.main()
