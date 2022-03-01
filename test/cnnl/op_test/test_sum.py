from __future__ import print_function

import sys
import os
import itertools
import unittest
import logging

import torch
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

# The sum operator uses the calculation result of double data type as the
# reference value, while the calculation error of float type is large.

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_sum_dim(self):
        type_list = [True, False]
        shape_list = [(1,32,5,12,8),(2,128,10,6),(2,512,8),(1,100),(24,),(2,0,3)]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len+1):
                dim_lists = list(itertools.permutations(range(dim_len), i)) + \
                    list(itertools.permutations(range(-dim_len, 0), i))
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        out_cpu = x.double().sum(test_dim, keepdim=test_type).float()
                        out_mlu = self.to_device(x).sum(test_dim, keepdim=test_type)
                        self.assertTensorsEqual(
                            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)


    # @unittest.skip("not test")
    @testinfo()
    def test_sum(self):
        shape_list = [(2,3,4,3,4,2,1),(2,3,4),(1,32,5,12,8),
                      (2,128,10,6),(2,512,8),(1,100),(24,),(2,0,3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.sum(x.double()).float()
            out_mlu = torch.sum(self.to_mlu(x))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sum_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.sum(x.double()).float()
        out_mlu = torch.sum(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sum_out(self):
        type_list = [True, False]
        shape_list = [(1,32,5,12,8),(2,128,10,6),(2,512,8),(1,100),(24,),(2,0,3)]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len+1):
                dim_lists = list(itertools.permutations(range(dim_len), i)) + \
                    list(itertools.permutations(range(-dim_len, 0), i))
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        out_cpu = torch.randn(1)
                        out_mlu = self.to_mlu(torch.randn(1))
                        x_mlu = self.to_mlu(x)
                        torch.sum(x.double(), test_dim, keepdim=test_type, out=out_cpu)
                        torch.sum(x_mlu, test_dim, keepdim=test_type, out=out_mlu)
                        try:
                            self.assertTensorsEqual(
                                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)
                        except AssertionError as e:
                            # results of CPU and MLU are out of threshold and
                            # MLU, numpy, gpu results are same, so use numpy
                            # result to compare with MLU
                            print(e)
                            # use double to ensure precision
                            x_numpy = x.double().numpy()
                            out_sum = np.sum(x_numpy, axis=test_dim, keepdims=test_type)
                            # np.sum returns float for full-dim sum,
                            # use torch.tensor instead of torch.from_numpy
                            if isinstance(out_sum, np.ndarray):
                                self.assertTensorsEqual(
                                    torch.from_numpy(out_sum).float(),
                                    out_mlu.cpu(), 0.003, use_MSE=True)
                            else:
                                self.assertTensorsEqual(
                                    torch.tensor(out_sum.item()),
                                    torch.tensor(out_mlu.cpu().item()), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sum_dtype(self):
        shape = (2,3,4)
        type_list = [torch.int, torch.short, torch.int8, torch.long, torch.uint8]
        for t in type_list:
            x = (torch.randn(shape, dtype=torch.float) * 10000).to(t)
            out_cpu = x.sum(dim=1, keepdim=True)
            out_mlu = x.to('mlu').sum(dim=1, keepdim=True)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)

        shape = (2,3,4)
        type_list = [torch.int, torch.short, torch.int8, torch.long, torch.uint8]
        for t in type_list:
            x = (torch.randn(shape, dtype=torch.float) * 10000).to(t)
            out_cpu = x.sum()
            out_mlu = x.to('mlu').sum()
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)

if __name__ == "__main__":
    unittest.main()
