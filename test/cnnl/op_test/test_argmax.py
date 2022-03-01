from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

class TestMaxOp(TestCase):

    def check_result(self, x, ind_cpu, ind_mlu, dim):
        # max sorting algorithm for mlu is different from cpu,
        # when the max result has multi-

        if dim is None:
            # None dim means index is one number for max-value in full storage.
            x_tr = x.view(-1)
            ind_cpu_tr = ind_cpu.view(-1)
            ind_mlu_tr = ind_mlu.view(-1)
            t = None
        else:
            # the follow transpose and reshape will move the dim(reduce dim)
            # to the first of shape, and reshape it as [dim_size, other_size]
            # and then the arange t will select max-value due to the index,
            # so we can check if mlu and cpu choose the same max-value.
            x_tr = x.transpose(dim, 0).reshape(x.shape[dim], -1)
            ind_cpu_tr = ind_cpu.transpose(dim, 0).reshape(1, -1)
            ind_mlu_tr = ind_mlu.transpose(dim, 0).reshape(1, -1)
            t = torch.arange(0, x_tr.shape[1])
        self.assertTensorsEqual(x_tr[ind_cpu_tr[0, t], t], x_tr[ind_mlu_tr[0, t], t],
                                0.0)

    #@unittest.skip("not test")
    @testinfo()
    def test_argmax(self):
        dtype_list = [torch.float, torch.int, torch.half]
        shape_list = [(2, 3, 4), (10, 11, 9, 8), (32,),
                      (15, 16, 8, 9, 10, 11), (2, 3, 4, 5, 6, 7, 8)]
        dim_list = [1, -1, 0, 2, 3, 6, 7, None]
        keepdim_choices = [True, False]
        mode_list = [self.to_non_dense, lambda x:x]
        list_list = [dtype_list, shape_list, dim_list, keepdim_choices, mode_list]
        for dtype, shape, dim, keepdim, mode in product(*list_list):
            x = torch.randn(shape)
            if dtype == torch.int:
                x = torch.randint(-10, 10, shape)
            x = x.to(dtype)
            if dim is not None and dim >= len(shape):
                dim = dim % len(shape)
            out_cpu = torch.argmax(mode(x), dim, keepdim=keepdim)
            out_mlu = torch.argmax(mode(self.to_device(x)), dim, keepdim=keepdim)
            out_mlu = out_mlu.cpu()
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertEqual(out_cpu.size(), out_mlu.size())
            if (not keepdim) and (dim is not None):
                out_cpu = out_cpu.unsqueeze(dim)
                out_mlu = out_mlu.unsqueeze(dim)
            if dtype == torch.half:
                x = x.to(torch.float)
            self.check_result(x, out_cpu, out_mlu, dim)

if __name__ == '__main__':
    unittest.main()
