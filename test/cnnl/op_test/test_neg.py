from __future__ import print_function

import sys
import os
import copy
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_neg(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        dtype_list = [torch.float, torch.half]
        err = 0.0
        for shape, dtype in product(shape_list, dtype_list):
            if dtype == torch.half:
                err = 3e-3
            x = torch.randn(shape, dtype=dtype)
            out_cpu = torch.neg(x)
            out_mlu = torch.neg(self.to_mlu_dtype(x, dtype))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_neg_exception(self):
        a = torch.randn(3).bool().to('mlu')
        ref_msg = "neg only support torch.float and torch.half."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.neg(a)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_channels_last(self):
        shape_list = [(2, 2, 512, 1024),]
        dtype_list = [torch.float, torch.half]
        err = 0.0
        for shape, dtype in product(shape_list, dtype_list):
            if dtype == torch.half:
                err = 3e-3
            x = torch.randn(shape, dtype=dtype).to(memory_format=torch.channels_last)
            out_cpu = torch.neg(x)
            out_mlu = torch.neg(self.to_mlu_dtype(x, dtype))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_inplace_contiguous(self):
        shape_list = [(512, 256, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            x.neg_()
            x_mlu.neg_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_inplace_channel_last(self):
        shape_list = [(512, 256, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            x_mlu = copy.deepcopy(x).to('mlu')
            x.neg_()
            x_mlu.neg_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_inplace_not_dense(self):
        shape_list = [(512, 256, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to('mlu')
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            x_mlu = copy.deepcopy(x).to('mlu')
            x.neg_()
            x_mlu.neg_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_out_contiguous(self):
        shape_list = [(512, 256, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (100), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to('mlu')
            out_cpu = torch.neg(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.neg(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            y = torch.randn((1))
            y_mlu = copy.deepcopy(y).to('mlu')
            out_cpu = torch.neg(x, out=y)
            out_mlu = torch.neg(self.to_mlu(x), out=y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_out_channel_last(self):
        shape_list = [(512, 256, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (100), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            y_mlu = copy.deepcopy(y).to('mlu')
            out_cpu = torch.neg(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.neg(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_out_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.to('mlu')
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            y = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            y_mlu = copy.deepcopy(y).to('mlu')
            out_cpu = torch.neg(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.neg(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_not_dense(self):
        shape_list = [(2, 2, 512, 1024),]
        dtype_list = [torch.float, torch.half]
        err = 0.0
        for shape, dtype in product(shape_list, dtype_list):
            if dtype == torch.half:
                err = 3e-3
            x = torch.randn(shape, dtype=dtype)
            out_cpu = torch.neg(x[..., :4])
            out_mlu = torch.neg(self.to_mlu_dtype(x, dtype)[..., :4])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

if __name__ == "__main__":
    unittest.main()
