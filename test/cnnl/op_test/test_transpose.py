from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestTransposeOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_transpose(self):
        shape_lst = [(126, 24, 1024), (4, 12, 45, 100), (4, 5, 6, 7, 8), (3, 4, 10, 200, 10, 20)]
        dim0_lst = [0, 1, 2]
        dim1_lst = [0, 1, 2]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for dim0 in dim0_lst:
                for dim1 in dim1_lst:
                    for data_type, err in data_types:
                        x = torch.randn(shape, dtype=torch.float)
                        x_mlu = self.to_mlu_dtype(x, data_type)
                        output_cpu = x.transpose(dim0, dim1)
                        output_mlu = x_mlu.transpose(dim0, dim1)
                        self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_transpose_channels_last(self):
        shape_lst = [(126, 24, 24, 30), (4, 12, 45, 20), (4, 5, 6, 7, 8)]
        dim0_lst = [0, 1, 2, 3]
        dim1_lst = [0, 1, 2, 3]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for dim0 in dim0_lst:
                for dim1 in dim1_lst:
                    for data_type, err in data_types:
                        if len(shape) == 4:
                            memory_type = torch.channels_last
                        elif len(shape) == 5:
                            memory_type = torch.channels_last_3d
                        else:
                            memory_type = torch.contiguous_format
                        x = torch.randn(shape, dtype=torch.float).to(
                            memory_format = memory_type)
                        x_mlu = self.to_mlu_dtype(x, data_type)
                        output_cpu = x.transpose(dim0, dim1)
                        output_mlu = x_mlu.transpose(dim0, dim1)
                        self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_transpose_not_dense(self):
        shape_lst = [(126, 24, 1024), (4, 12, 45, 100), (4, 5, 6, 7, 8)]
        dim0_lst = [0, 1, 2]
        dim1_lst = [0, 1, 2]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for dim0 in dim0_lst:
                for dim1 in dim1_lst:
                    for data_type, err in data_types:
                        x = torch.randn(shape, dtype=torch.float)
                        x_mlu = self.to_mlu_dtype(x, data_type)
                        output_cpu = x[:2,].transpose(dim0, dim1)
                        output_mlu = x_mlu[:2,].transpose(dim0, dim1)
                        self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_t(self):
        shape_lst = [(3, 44), (6, 123), (45, 100), (23), ()]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for data_type, err in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                output_cpu = x.t()
                output_mlu = x_mlu.t()
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)


if __name__ == "__main__":
    unittest.main()
