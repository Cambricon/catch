from __future__ import print_function

import sys
import logging
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import unittest
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411
logging.basicConfig(level=logging.DEBUG)

class TestIndexFillOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_index_fill(self):
        shape_list =[(0, 1, 2), (2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 0], [0, 2, 1], [0, 2, 1, 2, 1], [0, 1, 2], [0, 2, 2]]
        dim_list = [1, -2, 1, 2, 0]
        for i, shape in enumerate(shape_list):
            x = torch.randn(shape, dtype = torch.float)
            x_mlu = copy.deepcopy(x).to('mlu')
            index = torch.tensor(index_list[i])
            index_mlu = index.to('mlu')
            out_cpu = torch.index_fill(x, dim_list[i], index, 2)
            out_mlu = torch.index_fill(x_mlu, dim_list[i], index_mlu, 2)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_dtype(self):
        shape_list =[(2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3)]
        type_list = [torch.float, torch.int, torch.short, torch.int8, torch.int32, torch.uint8]
        for t in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype = torch.float).to(t)
                x_mlu = copy.deepcopy(x).to('mlu')
                index = torch.tensor([0, 2])
                index_mlu = index.to('mlu')
                x.index_fill_(1, index, 2)
                x_mlu.index_fill_(1, index_mlu, 2)
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_fill_memory_format(self):
        shape_list = [(3, 4, 5, 2, 3)]
        dim_list = [-1, 0, 1, 2, 3, 4]
        memory_formats = [True, False]
        list_list = [shape_list, dim_list, memory_formats]
        for shape, dim, channel_last in product(*list_list):
            x = torch.randn(shape, dtype = torch.float)
            if channel_last is True:
                x = self.convert_to_channel_last(x)
            x_mlu = copy.deepcopy(x).to('mlu')
            index = torch.tensor([1])
            index_mlu = index.to('mlu')
            x.index_fill_(dim, index, -1)
            x_mlu.index_fill_(dim, index_mlu, -1)
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_add_not_dense(self):
        shape_list =[(2, 4, 5), (4, 4, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1], [0, 1, 2], [0, 2, 2]]
        dim_list = [-2, 1, 2, 0]
        for i, shape in enumerate(shape_list):
            x = torch.randn(shape, dtype = torch.float)
            x_mlu = copy.deepcopy(x).to('mlu')
            index = torch.tensor(index_list[i])
            index_mlu = index.to('mlu')
            out_cpu = torch.index_fill(x[:,:3,...], dim_list[i], index, 1)
            out_mlu = torch.index_fill(x_mlu[:,:3,...], dim_list[i],
                      index_mlu, 1)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_add_inplace_not_dense(self):
        shape_list =[(2, 4, 5), (4, 4, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1], [0, 1, 2], [0, 2, 2]]
        dim_list = [-2, 1, 2, 0]
        for i, shape in enumerate(shape_list):
            x = torch.randn(shape, dtype = torch.float)
            x_mlu = copy.deepcopy(x).to('mlu')
            ori_ptr = x_mlu.data_ptr()
            index = torch.tensor(index_list[i])
            index_mlu = index.to('mlu')
            x[:,:3,...].index_fill_(dim_list[i], index, 2)
            x_mlu[:,:3,...].index_fill_(dim_list[i], index_mlu, 2)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_fill_value(self):
        shape_list = [(3, 4, 5, 2, 3)]
        for value in [-1, 0, 1, 1.2, -3.4]:
            for shape in shape_list:
                x = torch.randn(shape, dtype = torch.float)
                x_mlu = copy.deepcopy(x).to('mlu')
                index = torch.tensor([1])
                index_mlu = index.to('mlu')
                x.index_fill_(1, index, value)
                x_mlu.index_fill_(1, index_mlu, value)
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_fill_exception(self):
        shape =(2, 4, 5)
        x = torch.randn(shape, dtype = torch.float)
        x_mlu = copy.deepcopy(x).to('mlu')
        index = torch.randn(1, 2).to("mlu")

        ref_msg = "Expected tensor for argument #1 'input' to have one of the following scalar"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.to(torch.bool).index_fill_(1, torch.tensor([1]).to('mlu'), -1)

        ref_msg = "Expected object of scalar type Long but got scalar type Float for argument"
        ref_msg += " #3 'index' in call to _th_index_fill_"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.index_fill_(1, index, -1)

        ref_msg = "Expected object of scalar type Long but got scalar type Float for argument"
        ref_msg += " #3 'index' in call to _th_index_fill_"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.index_fill_(1, index.bool(), -1)

        index = torch.randn((shape)).int()
        ref_msg = "Index is supposed to be a vector"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.index_fill_(1, index, -1)

if __name__ == '__main__':
    unittest.main()
