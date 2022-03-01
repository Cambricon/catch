from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestLogOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_log(self):
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (24, 4560, 3, 6, 20),  (1), (0), ()]
        channel_first = [True, False]
        for shape in shape_list:
            for channel in channel_first:
                x = torch.rand(shape) + 0.0001
                out_cpu = torch.log(x)
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_mlu = torch.log(self.to_device(x))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().contiguous(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_log_inplace(self):
        torch.manual_seed(0)
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (1)]
        for shape in shape_list:
            x = torch.rand(shape) + 0.0001
            x = self.get_not_contiguous_tensor(x)
            x_mlu = self.to_device(x)
            x.log_()
            data_ptr_pre = x_mlu.data_ptr()
            x_mlu.log_()
            self.assertTensorsEqual(x, x_mlu.cpu().contiguous(), 0.003, use_MSE = True)
            self.assertEqual(data_ptr_pre, x_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_log_out(self):
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (1)]
        out_shape_list = [(24), (240, 8760), (1)]
        channel_first = [True, False]
        for shape, out_shape in zip(shape_list, out_shape_list):
            for channel in channel_first:
                x = torch.rand(shape) + 0.0001
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_cpu = torch.randn(out_shape)
                out_mlu = self.to_device(torch.randn(out_shape))
                torch.log(x, out=out_cpu)
                torch.log(self.to_device(x), out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log2(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape in shape_list:
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    out_cpu = torch.log2(x.float())
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_mlu = torch.log2(self.to_mlu_dtype(x, data_type))
                    self.assertTensorsEqual(out_cpu.float(), \
                        out_mlu.cpu().float().contiguous(), err, use_MSE=True)

                    x = torch.tensor([1.0017]).to(data_type)
                    out_cpu = torch.log2(x.float())
                    out_mlu = torch.log2(self.to_mlu_dtype(x, data_type))
                    self.assertTensorsEqual(out_cpu.float(), \
                        out_mlu.cpu().float().contiguous(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log2_inplace(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for data_type, err in dtype_list:
            for shape in shape_list:
                x = torch.rand(shape).to(data_type) + 1
                x = self.get_not_contiguous_tensor(x)
                x_mlu = self.to_mlu_dtype(x, data_type)
                x = x.float()
                x.log2_()
                data_ptr_pre = x_mlu.data_ptr()
                x_mlu.log2_()
                self.assertTensorsEqual(x.float(), \
                    x_mlu.cpu().float(), err, use_MSE=True)
                self.assertEqual(data_ptr_pre, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_log2_out(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        out_shape_list = [(12), (24), (240, 8760), (3, 2, 4), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape, out_shape in zip(shape_list, out_shape_list):
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_cpu = torch.randn(out_shape)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                    torch.log2(x.float(), out=out_cpu)
                    torch.log2(self.to_mlu_dtype(x, data_type), out=out_mlu)
                    self.assertTensorsEqual(out_cpu.float(), \
                        out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log10(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape in shape_list:
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    out_cpu = torch.log10(x.float())
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_mlu = torch.log10(self.to_mlu_dtype(x, data_type))
                    self.assertTensorsEqual(out_cpu.float(), \
                        out_mlu.cpu().float().contiguous(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log10_inplace(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for data_type, err in dtype_list:
            for shape in shape_list:
                x = torch.rand(shape).to(data_type) + 1
                x = self.get_not_contiguous_tensor(x)
                x_mlu = self.to_mlu_dtype(x, data_type)
                x = x.float()
                x.log10_()
                data_ptr_pre = x_mlu.data_ptr()
                x_mlu.log10_()
                self.assertTensorsEqual(x.float(), \
                    x_mlu.cpu().float(), err, use_MSE=True)
                self.assertEqual(data_ptr_pre, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_log10_out(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        out_shape_list = [(12), (24), (240, 8760), (3, 2, 4), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape, out_shape in zip(shape_list, out_shape_list):
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_cpu = torch.randn(out_shape)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                    torch.log10(x.float(), out=out_cpu)
                    torch.log10(self.to_mlu_dtype(x, data_type), out=out_mlu)
                    self.assertTensorsEqual(out_cpu.float(), \
                        out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log2_exception(self):
        x_mlu = torch.rand(1, 16).int().to('mlu')
        ref_msg = r"^log2 only supports floating-point dtypes for input, got\: Int$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.log2(x_mlu)
        ref_msg = r"^log2 only supports floating-point dtypes for input, got\: Int$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.log2_()
        out_mlu = torch.zeros(1).int().to('mlu')
        ref_msg = r"^log2 only supports floating-point dtypes for input, got\: Int$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.log2(x_mlu, out=out_mlu)
        x_mlu = torch.rand(1, 16).to('mlu')
        ref_msg = r"^log2 expected dtype Float but found Int$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.log2(x_mlu, out=out_mlu)

if __name__ == '__main__':
    unittest.main()
