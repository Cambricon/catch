from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch
import torch_mlu # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

class TestGatherOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_gather(self):
        shapes = [(32, 3, 224, 224), (2, 100, 56), (234, 32), (0, 32), (24,)]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                index = torch.abs(torch.rand(shape, dtype=torch.float)*shape[dim]).to(torch.int64)
                out = torch.gather(x, dim, index)
                x_mlu = self.to_mlu(x)
                index_mlu = self.to_device(index)
                out_mlu = torch.gather(x_mlu, dim, index_mlu)
                self.assertTensorsEqual(out, out_mlu.cpu().float(),
                                 0.000, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_out(self):
        shapes = [(32, 3, 224, 224), (2, 100, 56), (234, 32), (0, 32), (24,)]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                index = torch.abs(torch.rand(shape, dtype=torch.float)*shape[dim]).to(torch.int64)
                out_cpu = torch.randn(1)
                torch.gather(x, dim, index, out=out_cpu)
                x_mlu = x.to('mlu')
                index_mlu = index.to('mlu')
                out_mlu = torch.randn(1).to('mlu')
                torch.gather(x_mlu, dim, index_mlu, out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                 0.000, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_channelslast_and_nodense(self):
        def run_test(x, dim, index):
            out_cpu = torch.gather(x, dim, index)
            out_mlu = torch.gather(x.to('mlu'), dim, index.to('mlu'))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                    0.000, use_MSE = True)

        shapes = [(32, 3, 224, 224), (2, 100, 56, 56), (234, 3, 32, 32)]
        dims = [0, 1, 2, 3]
        for shape in shapes:
            for dim in dims:
                x = torch.randn(shape, dtype=torch.float)
                index = torch.abs(torch.rand(shape, dtype=torch.float)*shape[dim]).to(torch.int64)
                # channels_last input
                run_test(x.to(memory_format = torch.channels_last), dim, index)

                # not-dense input
                x = x[..., :2]
                shape = x.shape
                index = torch.randint(0, shape[dim], shape)
                run_test(x, dim, index)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_exception(self):
        shape = (2, 100, 56)
        dim = 0
        x = torch.randn(shape, dtype=torch.float)
        index = torch.randint(0, shape[dim], shape)
        x_mlu = self.to_mlu(x)
        index_mlu = self.to_device(index)
        ref_msg = "index dtype should be int/long"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(x_mlu, dim, index_mlu.float())
        ref_msg = "self and output 2 have unsame dim"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(x_mlu, dim, index_mlu.resize_(2, 100, 18))
        index_mlu = self.to_device(index)
        ref_msg = "self and index must have same dim"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(x_mlu, dim, index_mlu.resize_(100, 112))
        index_mlu = self.to_device(index)
        ref_msg = "self and output 1 have unsame dim"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(x_mlu, dim, index_mlu.resize_(100, 2, 56))

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_zero(self):
        s = torch.ones(size=[], dtype=torch.float)
        index = torch.zeros(size=(1,), dtype=torch.long)
        a = torch.gather(s, 0, index)
        b = torch.gather(s.to('mlu'), 0, index.to('mlu'))
        self.assertTensorsEqual(a, b.cpu(),
                                0.000, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_dtype(self):
        shape = (2, 100, 56)
        dim = 0
        dtype_list = [torch.double, torch.float, torch.half, torch.long, torch.int,
                      torch.short, torch.bool]
        for data_dtype in dtype_list:
            input = 100 * torch.rand(shape)
            input = input.to(data_dtype)
            index = torch.abs(torch.rand(shape, dtype=torch.float)*shape[dim]).to(torch.int64)
            out = torch.gather(input, dim, index)
            input_mlu = self.to_mlu(input)
            index_mlu = self.to_device(index)
            out_mlu = torch.gather(input_mlu, dim, index_mlu)
            self.assertTensorsEqual(out.double(), out_mlu.cpu().double(),
                                  0.000, use_MSE = True)

if __name__ == '__main__':
    unittest.main()
