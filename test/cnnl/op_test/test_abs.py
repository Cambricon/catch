from __future__ import print_function

import sys
import logging
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import unittest
import torch

import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestAbsOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_abs_contiguous(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (1000), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.abs(x)
            out_mlu = torch.abs(x.to('mlu'))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_channel_last(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (1000), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            out_cpu = torch.abs(x)
            out_mlu = torch.abs(x.to('mlu'))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 8), (10, 3, 32, 64), (2, 3, 8),
                      (254, 254, 112, 1, 1, 6)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.to(ct.mlu_device())
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            out_cpu = torch.abs(x)
            out_mlu = torch.abs(x.to('mlu'))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_contiguous(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (1000), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to(ct.mlu_device())
            out_cpu = torch.abs(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.abs(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_channel_last(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (1000), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            y_mlu = copy.deepcopy(y).to(ct.mlu_device())
            out_cpu = torch.abs(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.abs(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.to(ct.mlu_device())
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            y = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            y_mlu = copy.deepcopy(y).to(ct.mlu_device())
            out_cpu = torch.abs(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.abs(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_shape_contiguous(self):
        x = torch.randn(10000, dtype=torch.float)
        y = torch.randn(1000, dtype=torch.float)
        y_mlu = copy.deepcopy(y).to(ct.mlu_device())
        out_cpu = torch.abs(x, out=y)
        ori_ptr = y_mlu.data_ptr()
        out_mlu = torch.abs(self.to_mlu(x), out=y_mlu)
        out_ptr = y_mlu.data_ptr()
        assert ori_ptr != out_ptr
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        x = torch.randn(1000, dtype=torch.float)
        y = torch.randn(10000, dtype=torch.float)
        y_mlu = copy.deepcopy(y).to(ct.mlu_device())
        out_cpu = torch.abs(x, out=y)
        ori_ptr = y_mlu.data_ptr()
        out_mlu = torch.abs(self.to_mlu(x), out=y_mlu)
        out_ptr = y_mlu.data_ptr()
        self.assertEqual(ori_ptr, out_ptr)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_t_contiguous(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = x.abs()
            out_mlu = self.to_mlu(x).abs()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_t_channel_last(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            out_cpu = x.abs()
            out_mlu = self.to_mlu(x).abs()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_t_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.to(ct.mlu_device())
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            out_cpu = x.abs()
            out_mlu = self.to_mlu(x).abs()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_inplace_contiguous(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to(ct.mlu_device())
            x.abs_()
            x_mlu.abs_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_inplace_channel_last(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            x_mlu = copy.deepcopy(x).to(ct.mlu_device())
            x.abs_()
            x_mlu.abs_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_inplace_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to(ct.mlu_device())
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            x_mlu = copy.deepcopy(x).to(ct.mlu_device())
            x.abs_()
            x_mlu.abs_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_abs_exception(self):
        a = torch.randn(3).int().to('mlu')
        ref_msg = "Expected tensor for argument #1 'input' to have one of the following"
        ref_msg = ref_msg + " scalar types: Float, Half; but got MLUIntType instead"
        ref_msg = ref_msg + r" \(while checking arguments for abs\)"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.abs(a)

if __name__ == '__main__':
    unittest.main()
