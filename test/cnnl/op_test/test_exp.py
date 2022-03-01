from __future__ import print_function
import logging
import unittest
import sys
import os
import torch
os.environ['ENABLE_CNNL_TRYCATCH']='OFF'
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase # pylint: disable=C0413
logging.basicConfig(level=logging.DEBUG)

class TestExpOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_exp_contiguous(self):
        shape_list = [(16,384,3072), (16, 0, 88)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.exp(x)
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_channel_last(self):
        shape_list = [(2,3,3,4)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float).to(memory_format=torch.channels_last)
                out_cpu = torch.exp(x)
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_not_dense(self):
        shape_list = [(2,3,3,4)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.exp(x[:, :, :, :2])
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type)[:, :, :, :2])
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp_inplace_contiguous(self):
        shape_list = [(27), (13, 78), (16, 384, 3072), (13, 24, 35, 46), (16, 0, 88)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.exp_(x)
                torch.exp_(x_mlu)
                self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp_inplace_channel_last(self):
        shape_list = [(2,3,3,4)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float).to(memory_format=torch.channels_last)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.exp_(x)
                torch.exp_(x_mlu)
                self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp_inplace_not_dense(self):
        shape_list = [(2,3,3,4)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.exp_(x[:, :, :, :2])
                torch.exp_(x_mlu[:, :, :, :2])
                self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp_out(self):
        shape_list = [(27), (13, 78), (16, 384, 3072), (13, 24, 35, 46), (16, 0, 88)]
        data_types = [torch.float, torch.half]
        out_shapes = [(100, 10), (1), (20, 20, 60, 100), (77, 0, 88, 99)]
        for out_shape in out_shapes:
            for shape in shape_list:
                for data_type in data_types:
                    x = torch.randn(shape, dtype=torch.float)
                    x_mlu = self.to_mlu_dtype(x, data_type)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                    torch.exp(x, out=out_cpu)
                    torch.exp(x_mlu, out=out_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp_exception(self):
        x_mlu = torch.randn((16,384,3072)).int().to('mlu')
        ref_msg = "exp only support floating type"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.exp(x_mlu)

if __name__ == '__main__':
    unittest.main()
