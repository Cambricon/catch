from __future__ import print_function

import sys
import os
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413
logging.basicConfig(level=logging.DEBUG)

class TestAnyOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_any_dim(self):
        shape_list = [(10,), (3, 5), (4, 5, 8), (8, 10, 12, 14),
                      (0,), (0, 5), (4, 5, 0), (8, 0, 12, 14)]
        dim_list = [0, 1, -1, 3,
                    0, 1, -1, 3]
        for shape, dim in zip(shape_list, dim_list):
            x = torch.rand(shape, dtype=torch.float)
            if x.dim() == 4:
                x = x.to(memory_format = torch.channels_last)
            x_1 = x < 0.05
            out_cpu_1 = x_1.any()
            out_cpu_2 = x_1.any(dim)
            out_mlu_1 = self.to_mlu(x_1).any()
            out_mlu_2 = self.to_mlu(x_1).any(dim)
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right")
            # Bool Result diff: 0.0
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_mlu_2.dtype == out_cpu_2.dtype, "any out dtype is not right")
            # Bool Result diff: 0.0

    # @unittest.skip("not test")
    @testinfo()
    def test_any(self):
        shape_list = [(10,), (3, 5), (4, 5, 8), (8, 10, 12, 14),
                      (0,), (0, 5), (4, 5, 0), (8, 0, 12, 14)]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float)
            if x.dim() == 4:
                x = x.to(memory_format = torch.channels_last)
            x_1 = x < 0.05
            out_cpu_1 = x_1.any()
            out_mlu_1 = self.to_mlu(x_1).any()
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right")
            # Bool Result diff: 0.0

            self.assertTrue(out_cpu_1.size() == out_mlu_1.size())
            self.assertTrue(out_cpu_1.stride() == out_mlu_1.stride())
        x_1 = torch.tensor(True)
        out_cpu_1 = x_1.any()
        out_mlu_1 = self.to_mlu(x_1).any()
        self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
        self.assertTrue(out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right")
        x_1 = torch.tensor(False)
        out_cpu_1 = x_1.any()
        out_mlu_1 = self.to_mlu(x_1).any()
        self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
        self.assertTrue(out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right")
      
    # @unittest.skip("not test")
    @testinfo()
    def test_any_not_contiguous(self):
        shape_list = [(100, 200), (99, 30, 40),
                      (34, 56, 78, 90)]
        dim_list = [-2, 1, 2]
        for i, list_ in enumerate(shape_list):
            x = torch.rand(list_, dtype=torch.float)
            if x.dim() == 4:
                x = x.to(memory_format = torch.channels_last)
            x_1 = x.round().bool()
            out_cpu_1 = x_1[:,1:].any()
            out_cpu_2 = x_1[:,1:].any(dim_list[i])
            out_mlu_1 = self.to_mlu(x_1)[:,1:].any()
            out_mlu_2 = self.to_mlu(x_1)[:,1:].any(dim_list[i])
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.003, use_MSE=True)  
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(), 0.003, use_MSE=True)  
            self.assertTrue(out_cpu_1.size() == out_mlu_1.size())
            self.assertTrue(out_cpu_2.size() == out_mlu_2.size())
            self.assertTrue(out_cpu_1.stride() == out_mlu_1.stride())
            self.assertTrue(out_cpu_2.stride() == out_mlu_2.stride())
    


if __name__ == '__main__':
    unittest.main()
