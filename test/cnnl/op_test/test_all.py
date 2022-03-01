from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411
logging.basicConfig(level=logging.DEBUG)

class TestAllOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_all_dim(self):
        shape_list = [(10, 11, 20), (1111, ), (2, 3, 4, 8, 10),
                      (34, 56, 78, 90), (), (0, 6)]
        dim_list = [-2, 0, -1, 3, 0, 1]
        dtype_list = [torch.bool, torch.uint8]
        keep_type = [True, False]
        for i, list_ in enumerate(shape_list):
            for dtype in dtype_list:
                for keep in keep_type:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    x_mlu = self.to_device(x_cpu)
                    out_cpu = x_cpu.all(dim=dim_list[i], keepdim=keep)
                    out_mlu = x_mlu.all(dim=dim_list[i], keepdim=keep)
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),\
                                            0.0, use_MSE=True)
                    self.assertTrue(out_mlu.dtype == out_cpu.dtype, "all out dtype is not right")
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

        # not contiguous
        shape_list_nc = [(10,11,20),(2,3,4,8,10), (34, 56, 78, 90), (0, 6)]
        dim_list_nc = [-2, -1, 3, 1]
        dtype_list_nc = [torch.bool, torch.uint8]
        keep_type_nc = [True, False]
        for i, list_ in enumerate(shape_list_nc):
            for dtype in dtype_list_nc:
                for keep in keep_type_nc:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    if x_cpu.dim() == 4:
                        x_cpu = x_cpu.to(memory_format = torch.channels_last)
                    x_mlu = self.to_device(x_cpu)
                    out_cpu = x_cpu[:,1:].all(dim=dim_list_nc[i], keepdim=keep)
                    out_mlu = x_mlu[:,1:].all(dim=dim_list_nc[i], keepdim=keep)
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),\
                                            0.0, use_MSE=True)
                    self.assertTrue(out_mlu.dtype == out_cpu.dtype, "all out dtype is not right")
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_all_exception(self):
        shape = (10, 20)
        x = torch.rand(shape, dtype=torch.float)
        ref_msg = "all only supports torch.uint8 and torch.bool dtypes"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            self.to_device(x).all()

        x = torch.rand(shape, dtype=torch.float).bool()
        out = torch.rand(shape, dtype=torch.float)
        ref_msg = "out and self should have same dtype"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.all(self.to_device(x), dim=0, keepdim=False, out=self.to_device(out))

    # @unittest.skip("not test")
    @testinfo()
    def test_all(self):
        shape_list = [(10, 11, 20), (1111, ), (2, 3, 4, 8, 10),
                      (34, 56, 78, 90), (), (0, 6)]
        dtype_list = [torch.bool, torch.uint8]
        for list_ in shape_list:
            for dtype in dtype_list:
                x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                x_mlu = self.to_device(x_cpu)
                out_cpu = x_cpu.all()
                out_mlu = x_mlu.all()
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),\
                                        0.0, use_MSE=True)
                self.assertTrue(out_mlu.dtype == out_cpu.dtype, "all out dtype is not right")
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

        # not contiguous
        shape_list_nc = [(10,11,20),(2,3,4,8,10), (34, 56, 78, 90), (0, 6)]
        dtype_list_nc = [torch.bool, torch.uint8]
        for list_ in shape_list_nc:
            for dtype in dtype_list_nc:
                x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                if x_cpu.dim() == 4:
                    x_cpu = x_cpu.to(memory_format = torch.channels_last)
                x_mlu = self.to_device(x_cpu)
                out_cpu = x_cpu[:,1:].all()
                out_mlu = x_mlu[:,1:].all()
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),\
                                        0.0, use_MSE=True)
                self.assertTrue(out_mlu.dtype == out_cpu.dtype, "all out dtype is not right")
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_all_out(self):
        shape_list = [(10, 11, 20), (1111, ), (2, 3, 4, 8, 10),
                      (34, 56, 78, 90), (), (0, 6)]
        dim_list = [-2, 0, -1, 3, 0, 1]
        dtype_list = [torch.bool, torch.uint8]
        keep_type = [True, False]
        for dim_, list_ in zip(dim_list, shape_list):
            for dtype in dtype_list:
                for keep in keep_type:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    out_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    x_mlu = self.to_device(x_cpu)
                    out_mlu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    out_mlu = self.to_device(out_mlu)
                    torch.all(x_cpu, dim=dim_, keepdim=keep, out=out_cpu)
                    torch.all(x_mlu, dim=dim_, keepdim=keep, out=out_mlu)
                    self.assertTrue(out_mlu.dtype == out_cpu.dtype, "all out dtype is not right")
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),\
                                            0.0, use_MSE=True)
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

        # not contiguous
        shape_list_nc = [(10,11,20),(2,3,4,8,10), (34, 56, 78, 90), (0, 6)]
        dim_list_nc = [-2, -1, 3, 1]
        dtype_list_nc = [torch.bool, torch.uint8]
        keep_type_nc = [True, False]
        for dim_, list_ in zip(dim_list_nc, shape_list_nc):
            for dtype in dtype_list_nc:
                for keep in keep_type_nc:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    out_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)[:,1:]
                    x_mlu = self.to_device(x_cpu)
                    out_mlu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)[:,1:]
                    out_mlu = self.to_device(out_mlu)
                    torch.all(x_cpu[:,1:], dim=dim_, keepdim=keep, out=out_cpu)
                    torch.all(x_mlu[:,1:], dim=dim_, keepdim=keep, out=out_mlu)
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),\
                                            0.0, use_MSE=True)
                    self.assertTrue(out_mlu.dtype == out_cpu.dtype, "all out dtype is not right")
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

if __name__ == '__main__':
    unittest.main()
