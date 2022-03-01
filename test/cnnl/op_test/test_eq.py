from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=all
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0411, C0413

logging.basicConfig(level=logging.DEBUG)

class TestEqOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_eq(self):
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
                out_cpu = torch.eq(x, y)
                out_mlu = torch.eq(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = x == y
                out_mlu = self.to_mlu(x) == self.to_mlu(y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_eq_channels_last(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((2, 3, 24, 30), (1, 1, 1, 30)),
                                   ((16, 8, 8, 32), (16, 8, 8, 32))]:
                x = torch.randn(shape1).to(t).to(memory_format = torch.channels_last)
                y = torch.randn(shape2).to(t).to(memory_format = torch.channels_last)
                out_cpu = torch.eq(x, y)
                out_mlu = torch.eq(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = x == y
                out_mlu = self.to_mlu(x) == self.to_mlu(y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                # mixed memory format
                z = torch.randn(shape2).to(t)
                out_cpu = torch.eq(x, z)
                out_mlu = torch.eq(self.to_mlu(x), self.to_mlu(z))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = x == z
                out_mlu = self.to_mlu(x) == self.to_mlu(z)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_not_dense(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((2, 3, 24, 30), (1, 1, 1, 30)),
                                   ((16, 8, 8, 32), (16, 8, 8, 32))]:
                x = torch.randn(shape1).to(t)[:, :, :, :15]
                y = torch.randn(shape2).to(t)[:, :, :, :15]
                out_cpu = torch.eq(x, y)
                out_mlu = torch.eq(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = x == y
                out_mlu = self.to_mlu(x) == self.to_mlu(y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_inplace(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((), ()), ((), (1)),
                                   ((), (256, 144, 7, 15, 2, 1)), ((1), (256, 7)),
                                   ((5), (5)),
                                   ((1), (256, 144, 7, 15, 2, 1, 1, 1))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y_mlu_data = y_mlu.data_ptr()
                y.eq_(x)
                y_mlu.eq_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_inplace_channels_last(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((1, 1, 1, 30), (2, 3, 24, 30)),
                                   ((16, 8, 8, 32), (16, 8, 8, 32))]:
                x = torch.randn(shape1).to(t).to(memory_format = torch.channels_last)
                y = torch.randn(shape2).to(t).to(memory_format = torch.channels_last)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y_mlu_data = y_mlu.data_ptr()
                y.eq_(x)
                y_mlu.eq_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_inplace_not_dense(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((1, 1, 1, 30), (2, 3, 24, 30)),
                                   ((16, 8, 8, 32), (16, 8, 8, 32))]:
                x = torch.randn(shape1).to(t)[:, :, :, :15]
                y = torch.randn(shape2).to(t)[:, :, :, :15]
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y_mlu_data = y_mlu.data_ptr()
                y.eq_(x)
                y_mlu.eq_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_out(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half
        ]
        for t in type_list:
            for shape1, shape2 in [((), ()), ((), (1)),
                                   ((), (25, 14, 7, 15, 2, 1)), ((1), (256, 7)),
                                   ((5), (5)),
                                   ((2, 3, 4), (3, 4)),
                                   ((25, 14, 7, 15, 2, 1, 1, 1), (1)),
                                   ((1), (25, 14, 7, 15, 2, 1, 1, 1))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2, dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2, dtype=torch.bool).to("mlu")
                torch.eq(x, y, out=out_tmpcpu)
                torch.eq(self.to_mlu(x), self.to_mlu(y), out=out_tmpmlu)
                self.assertTensorsEqual(out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_eq_scalar(self):
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
                out_cpu = torch.eq(x, y)
                out_mlu = torch.eq(self.to_mlu(x), y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = x == y
                out_mlu = self.to_mlu(x) == y
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_eq_inplace_scalar(self):
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
                x_mlu_data = x_mlu.data_ptr()
                x.eq_(y)
                x_mlu.eq_(y)
                self.assertEqual(x_mlu_data, x_mlu.data_ptr())
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_eq_out_scalar(self):
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
                torch.eq(x, y, out=out_tmpcpu)
                torch.eq(self.to_mlu(x), y, out=out_tmpmlu)
                self.assertTensorsEqual(
                    out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_exception(self):
        a = torch.randn(3).float().to('mlu')
        b = torch.randn(3).int().to('mlu')
        ref_msg = r"^Expected object of scalar type float but got scalar"
        ref_msg = ref_msg + r" type int for argument \'other\'$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.eq_(b)


if __name__ == '__main__':
    unittest.main()
