from __future__ import print_function

import sys
import os
import unittest
import logging
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0411, C0413

logging.basicConfig(level=logging.DEBUG)

class TestLtOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_lt(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((), ()), ((), (1)),
                                   ((5), (5)),
                                   ((2, 3, 4), (3, 4)),
                                   ((1, 11, 1, 4, 1, 5, 1, 2), (11, 1, 5, 1, 5, 1, 3, 1)),
                                   ((25, 14, 7, 15, 2, 1, 1, 1), (1))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu, y_mlu = self.to_mlu(x), self.to_mlu(y)
                if t == torch.half:
                    x, y = x.float(), y.float()
                out_cpu = torch.lt(x, y)
                out_mlu = torch.lt(x_mlu, y_mlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = x < y
                out_mlu = x_mlu < y_mlu
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_channel_last(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((5, 3, 4, 1), (1, 3, 4, 1))]:
                # both channel_last
                x = torch.randn(shape1).to(t).to(memory_format=torch.channels_last)
                y = torch.randn(shape2).to(t).to(memory_format=torch.channels_last)
                out_cpu = torch.lt(x, y)
                out_mlu = torch.lt(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = x < y
                out_mlu = self.to_mlu(x) < self.to_mlu(y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                # mixed memory format
                z = torch.randn(shape2).to(t)
                out_cpu = torch.lt(x, z)
                out_mlu = torch.lt(self.to_mlu(x), self.to_mlu(z))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = x < z
                out_mlu = self.to_mlu(x) < self.to_mlu(z)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_not_dense(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((2, 3, 4), (3, 4))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.lt(x[:, :, :2], y[:, :2])
                out_mlu = torch.lt(self.to_mlu(x)[:, :, :2], self.to_mlu(y)[:, :2])
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = x[:, :, :2] < y[:, :2]
                out_mlu = self.to_mlu(x)[:, :, :2] < self.to_mlu(y)[:, :2]
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_inplace(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((), ()), ((), (1)),
                                   ((), (256, 144, 7, 15, 2, 1)),
                                   ((1), (256, 7)),
                                   ((5), (5)),
                                   ((1), (256, 144, 7, 15, 2, 1, 1, 1))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu, y_mlu = x.to("mlu"), y.to("mlu")
                if t == torch.half:
                    x, y = x.float(), y.float()
                y_mlu_data = y_mlu.data_ptr()
                y.lt_(x)
                y_mlu.lt_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_inplace_channel_last(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.long, torch.half, torch.uint8
        ]
        for t in type_list:
            for shape1, shape2 in [((5, 3, 4, 1), (1, 3, 4, 1))]:
                # both channel last
                x = torch.randn(shape2).to(t).to(memory_format=torch.channels_last)
                y = torch.randn(shape1).to(t).to(memory_format=torch.channels_last)
                x_mlu, y_mlu = x.to("mlu"), y.to("mlu")
                if t == torch.half:
                    x, y = x.float(), y.float()
                y.lt_(x)
                y_mlu_data = y_mlu.data_ptr()
                y_mlu.lt_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)
                # mixed memory format
                z = torch.randn(shape1).to(t)
                z_mlu = z.to("mlu")
                if t == torch.half:
                    z = z.float()
                z.lt_(x)
                z_mlu_data = z_mlu.data_ptr()
                z_mlu.lt_(x_mlu)
                self.assertEqual(z_mlu_data, z_mlu.data_ptr())
                self.assertTensorsEqual(z.float(), z_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_inplace_not_dense(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.long, torch.uint8, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((3, 4), (2, 3, 4))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu, y_mlu = x.to("mlu"), y.to("mlu")
                if t == torch.half:
                    x, y = x.float(), y.float()
                y[:, :, :2].lt_(x[:, :2])
                y_mlu_data = y_mlu.data_ptr()
                y_mlu[:, :, :2].lt_(x_mlu[:, :2])
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_out(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((), ()), ((), (1)),
                                   ((), (25, 14, 7, 15, 2, 1)),
                                   ((1), (256, 7)),
                                   ((5), (5)),
                                   ((2, 3, 4), (3, 4)),
                                   ((25, 14, 7, 15, 2, 1, 1, 1), (1)),
                                   ((1), (25, 14, 7, 15, 2, 1, 1, 1))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2, dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2, dtype=torch.bool).to("mlu")
                x_mlu, y_mlu = self.to_mlu(x), self.to_mlu(y)
                if t == torch.half:
                    x, y = x.float(), y.float()
                torch.lt(x, y, out=out_tmpcpu)
                torch.lt(x_mlu, y_mlu, out=out_tmpmlu)
                self.assertTensorsEqual(
                    out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_scalar(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape in [(), (256, 144, 7, 15, 2, 1), (1), (256, 7),
                          (2, 3, 4), (117, 1, 5, 1, 5, 1, 3, 1),
                          (256, 144, 7, 15, 2, 1, 1, 1)]:
                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                x_mlu = self.to_mlu(x)
                if t == torch.half:
                    x = x.float()

                out_cpu = torch.lt(x, y)
                out_mlu = torch.lt(x_mlu, y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = x < y
                out_mlu = x_mlu < y
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_inplace_scalar(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape in [(), (256, 144, 7, 15, 2, 1), (1,), (256, 7),
                          (2, 3, 4), (117, 1, 5, 1, 5, 1, 3, 1),
                          (256, 144, 7, 15, 2, 1, 1, 1)]:

                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                x_mlu = x.to('mlu')
                if t == torch.half:
                    x = x.float()
                x_mlu_data = x_mlu.data_ptr()
                x.lt_(y)
                x_mlu.lt_(y)
                self.assertEqual(x_mlu_data, x_mlu.data_ptr())
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_out_scalar(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape in [(), (256, 144, 7, 15, 2, 1), (1), (256, 7),
                          (2, 3, 4), (117, 1, 5, 1, 5, 1, 3, 1),
                          (256, 144, 7, 15, 2, 1, 1, 1)]:
                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                out_tmpcpu = torch.zeros(shape, dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape, dtype=torch.bool).to('mlu')
                x_mlu = self.to_mlu(x)
                if t == torch.half:
                    x = x.float()
                torch.lt(x, y, out=out_tmpcpu)
                torch.lt(x_mlu, y, out=out_tmpmlu)
                self.assertTensorsEqual(
                    out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_not_dense(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((12, 15, 18, 26), (12, 15, 18, 50)),
                                   ((1), (20, 144, 8, 30))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu, y_mlu = self.to_mlu(x), self.to_mlu(y)
                if t == torch.half:
                    x, y = x.float(), y.float()
                out_cpu = torch.lt(x, y[:,:,:,10:36])
                out_mlu = torch.lt(x_mlu, y_mlu[:,:,:,10:36])
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = x < y[:,:,:,10:36]
                out_mlu = x_mlu < y_mlu[:,:,:,10:36]
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lt_channel_last(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1 in [(12, 15, 18, 26),
                           (20, 144, 8, 30)]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape1).to(t)
                x_cl = self.convert_to_channel_last(x)
                y_cl = self.convert_to_channel_last(y)
                x_mlu, y_mlu = self.to_mlu(x_cl), self.to_mlu(y_cl)
                if t == torch.half:
                    x_cl, y_cl = x_cl.float(), y_cl.float()

                out_cpu = torch.lt(x_cl, y_cl)
                x_mlu_cl = self.convert_to_channel_last(x_mlu)
                y_mlu_cl = self.convert_to_channel_last(y_mlu)
                out_mlu = torch.lt(x_mlu_cl, y_mlu_cl)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = x_cl < y_cl
                out_mlu = x_mlu_cl < y_mlu_cl
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
