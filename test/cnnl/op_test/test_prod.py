from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_prod_dtype_int32(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100)]

        x = torch.tensor(5, dtype=torch.int32)
        out_cpu = torch.prod(x)
        out_mlu = torch.prod(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=True)

        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randint(0, 3, shape)
                out_cpu = x.prod(item[1], keepdim=item[0])
                out_mlu = self.to_device(x).prod(item[1], keepdim=item[0])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True)

                x = torch.randint(0, 3, shape)
                out_cpu = torch.prod(x)
                out_mlu = torch.prod(self.to_device(x))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_dim(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100),(24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = x.prod(item[1], keepdim=item[0])
                out_mlu = self.to_device(x).prod(item[1], keepdim=item[0])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)


    # @unittest.skip("not test")
    @testinfo()
    def test_prod(self):
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100),(24,)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.prod(x)
            out_mlu = torch.prod(self.to_device(x))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_channel_last(self):
        x = torch.randn((2, 128, 10, 6), dtype=torch.float).to(memory_format = torch.channels_last)
        out_cpu = torch.prod(x)
        out_mlu = torch.prod(self.to_device(x))
        self.assertTensorsEqual(
            out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

        # test not dense
        out_cpu = torch.prod(x[..., :2])
        out_mlu = torch.prod(self.to_device(x)[..., :2])
        self.assertTensorsEqual(
            out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_scalar(self):
        x = torch.tensor(5.2, dtype = torch.float)
        out_cpu = torch.prod(x)
        out_mlu = torch.prod(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_out(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100),(24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.randn(1)
                out_mlu = self.to_device(out_cpu)
                x_mlu = self.to_device(x)
                torch.prod(x, item[1], keepdim=item[0], out=out_cpu)
                torch.prod(x_mlu, item[1], keepdim=item[0], out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_empty(self):
        x = torch.randn(1, 0, 1)
        out_mlu = x.to('mlu').prod()
        out_cpu = x.prod()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

if __name__ == "__main__":
    unittest.main()
