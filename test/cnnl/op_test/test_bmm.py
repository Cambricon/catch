from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import logging
import unittest
from itertools import product
import torch
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413

class TestBmmOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_bmm(self):
        dtype_list = [(torch.float, 3e-3)]
        shape_ab_list = [((3, 4, 5), (3, 5, 6)), ((4, 3, 5), (4, 5, 6)),
                         ((256, 10, 64), (256, 64, 10)), ((256, 10, 10), (256, 10, 64)),
                         ((0, 4, 5), (0, 5, 6)), ((3, 0, 4), (3, 4, 5)),
                         ((3, 4, 0), (3, 0, 6)), ((3, 4, 5), (3, 5, 0))]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x:x]
        for (data_type, err), (shape_a, shape_b), mode\
                in product(dtype_list, shape_ab_list, mode_list):
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type))
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type))
            out_cpu = torch.bmm(a, b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bmm_trans(self):
        dtype_list = [(torch.float, 3e-3)]
        shape_ab_list = [((3, 4, 5), (3, 5, 6)), ((4, 3, 5), (4, 5, 6)),
                         ((256, 10, 64), (256, 64, 10)), ((256, 10, 10), (256, 10, 64))]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x:x]
        for (data_type, err), (shape_a, shape_b), mode\
                in product(dtype_list, shape_ab_list, mode_list):
            # trans self
            a = torch.randn(shape_a, dtype=torch.float).transpose(1, 2).contiguous()
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).transpose(1, 2)
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type))
            out_cpu = torch.bmm(a.transpose(1, 2), b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # trans other
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float).transpose(1, 2).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type))
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).transpose(1, 2)
            out_cpu = torch.bmm(a, b.transpose(1, 2))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            #trans self and other
            a = torch.randn(shape_a, dtype=torch.float).transpose(1, 2).contiguous()
            b = torch.randn(shape_b, dtype=torch.float).transpose(1, 2).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).transpose(1, 2)
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).transpose(1, 2)
            out_cpu = torch.bmm(a.transpose(1, 2), b.transpose(1, 2))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bmm_permute(self):
        dtype_list = [(torch.float, 3e-3)]
        shape_ab_list = [((3, 4, 5), (3, 5, 6)), ((4, 3, 5), (4, 5, 6)),
                         ((256, 10, 64), (256, 64, 10)), ((256, 10, 10), (256, 10, 64)),
                         ((23, 23, 23), (23, 23, 23))]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x:x]
        for (data_type, err), (shape_a, shape_b), mode\
                in product(dtype_list, shape_ab_list, mode_list):
            # permute self
            a = torch.randn(shape_a, dtype=torch.float).permute(1, 2, 0).contiguous()
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).permute(2, 0, 1)
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type))
            out_cpu = torch.bmm(a.permute(2, 0, 1), b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # permute other
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float).permute(2, 1, 0).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type))
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).permute(2, 1, 0)
            out_cpu = torch.bmm(a, b.permute(2, 1, 0))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            #permute self and other
            a = torch.randn(shape_a, dtype=torch.float).permute(1, 0, 2).contiguous()
            b = torch.randn(shape_b, dtype=torch.float).permute(2, 0, 1).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).permute(1, 0, 2)
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).permute(1, 2, 0)
            out_cpu = torch.bmm(a.permute(1, 0, 2), b.permute(1, 2, 0))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
