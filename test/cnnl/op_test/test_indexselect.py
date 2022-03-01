from __future__ import print_function

import sys
import os
from itertools import product
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestIndexSelectOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_indexselect(self) :
        shape_list =[(2, 4, 5), (8, 9, 10), (8, 9, 10, 11), (3, 4, 5, 6, 7),
            (8, 9, 10, 11, 12, 14), (8, 9, 10, 11, 12, 13, 14), (99, 30, 40)]
        c_lists =[9796, 10, 8767]
        index_shape_list = [(3,) , (0,), ()]
        type_list = [torch.half, torch.float, torch.uint8, torch.long, torch.double,
                     torch.int, torch.short, torch.bool]
        for shape, t, index_shape in product(shape_list, type_list, index_shape_list):
            x = torch.randn(shape, dtype = torch.float).to(t)
            index = torch.randint(0, 3, (index_shape))
            for dim in[1]:
                out_cpu = torch.index_select(x, dim, index)
                out_mlu = torch.index_select(self.to_mlu(x), dim, index.to(ct.mlu_device()))
                self.assertTensorsEqual(out_cpu.float(),
                    out_mlu.cpu().float(), 0.0)
        #size use in transformer
        for c in c_lists:
            for dim in[0]:
                x = torch.rand(c, 512, dtype = torch.float)
                index = torch.randint(0, c, [320], dtype = torch.int)
                out_cpu = torch.index_select(x, dim, index.long())
                out_mlu = torch.index_select(self.to_mlu(x), dim, index.long().to(ct.mlu_device()))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_indexselect_not_dense(self) :
        shape_list =[(8, 9, 10, 20), (3, 10, 5, 6, 40)]
        for shape in shape_list:
            x0 = torch.randn(shape, dtype = torch.float)
            x = x0[:,:,:,10:16]
            x_mlu = self.to_mlu(x0)
            index = torch.tensor([1, 3, 2, 1, 2, 1, 4, 3, 5])
            for dim in[1]:
                out_cpu = torch.index_select(x, dim, index[2:5])
                index_mlu = index.to(ct.mlu_device())
                out_mlu = torch.index_select(x_mlu[:,:,:,10:16], dim, index_mlu[2:5])
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_indexselect_channel_last(self) :
        shape_list =[(8, 9, 10, 20), (3, 4, 5, 6, 40)]
        type_list = [torch.float, torch.long]
        for shape in shape_list:
            for dtype in type_list:
                x0 = torch.randn(shape).to(dtype = dtype)
                x = self.convert_to_channel_last(x0)
                x_mlu0 = self.to_mlu(x.clone())
                x_mlu = self.convert_to_channel_last(x_mlu0)
                index = torch.tensor([1, 3, 2])
                for dim in[1]:
                    out_cpu = torch.index_select(x, dim, index)
                    out_mlu = torch.index_select(x_mlu, dim, index.to(ct.mlu_device()))
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)


class TestIndexSelectOutOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_indexselect_out(self) :
        shape_list =[(2, 4, 5), (8, 9, 10, 11),
            (8, 9, 10, 11, 12, 14)]
        c_lists =[9796, 10, 8767]
        type_list = [torch.half, torch.float,
                torch.int, torch.short, torch.bool]
        for shape in shape_list:
            for t in type_list:
                x = torch.randn(shape, dtype = torch.float).to(t)
                out_cpu = torch.randn(1, dtype = torch.float).to(t)
                out_mlu = torch.randn(1, dtype = torch.float).to(t).to('mlu')
                index = torch.tensor([1, 3, 2])
                for dim in[1]:
                    torch.index_select(x, dim, index, out=out_cpu)
                    torch.index_select(self.to_mlu(x), dim, index.to(ct.mlu_device()), out=out_mlu)
                    self.assertTensorsEqual(out_cpu.float(),
                        out_mlu.cpu().float(), 0.0)
        for shape in shape_list:
            x = torch.randn(shape, dtype = torch.float)
            out_cpu = torch.randn(1, dtype = torch.float)
            out_mlu = torch.randn(1, dtype = torch.float).to('mlu')
            index = torch.tensor([1, 3, 2])
            for dim in[1]:
                torch.index_select(x, dim, index, out=out_cpu)
                torch.index_select(self.to_mlu(x), dim, index.to(ct.mlu_device()), out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)
            #size use in transformer
            for c in c_lists:
                for dim in[0]:
                    x = torch.rand(c, 512, dtype = torch.float)
                    index = torch.randint(0, c, [320], dtype = torch.int)
                    out_cpu = torch.index_select(x, dim, index.long())
                    out_mlu = torch.index_select(self.to_mlu(x), dim,
                        index.long().to(ct.mlu_device()))
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_indexselect_out_not_dense(self) :
        shape_list =[(8, 9, 10, 20), (3, 10, 5, 6, 40)]
        out_cpu = torch.randn(1, dtype = torch.float)
        out_mlu = torch.randn(1, dtype = torch.float).to('mlu')
        for shape in shape_list:
            x0 = torch.randn(shape, dtype = torch.float)
            x = x0[:,:,:,10:16]
            x_mlu = self.to_mlu(x0)
            index = torch.tensor([1, 3, 2, 1, 2, 1, 4, 3, 5])
            for dim in[1]:
                torch.index_select(x, dim, index[2:5], out=out_cpu)
                index_mlu = index.to(ct.mlu_device())
                torch.index_select(x_mlu[:,:,:,10:16], dim, index_mlu[2:5], out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_indexselect_out_channel_last(self) :
        shape_list =[(8, 9, 10, 20), (3, 4, 5, 6, 40)]
        out_cpu = torch.randn(1, dtype = torch.float)
        out_mlu = torch.randn(1, dtype = torch.float).to('mlu')
        for shape in shape_list:
            x0 = torch.randn(shape, dtype = torch.float)
            x = self.convert_to_channel_last(x0)
            x_mlu0 = self.to_mlu(x.clone())
            x_mlu = self.convert_to_channel_last(x_mlu0)
            index = torch.tensor([1, 3, 2])
            for dim in[1]:
                torch.index_select(x, dim, index, out=out_cpu)
                torch.index_select(x_mlu, dim, index.to(ct.mlu_device()), out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)


if __name__ == '__main__':
    unittest.main()
