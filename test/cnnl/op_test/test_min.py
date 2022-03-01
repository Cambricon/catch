from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411
logging.basicConfig(level=logging.DEBUG)

class TestMinOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_min_dim(self):
        shape_list = [(2,3,4), (1, 3, 224), (1, 3, 1, 1, 1), (1, 3, 224, 224)]
        dim_list = [1, -1, 0, 2]
        type_list = [True,False,True,False]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.min(x, dim_list[i], keepdim=type_list[i])
            out_mlu = torch.min(self.to_mlu(x), dim_list[i],keepdim=type_list[i])
            self.assertTensorsEqual(out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True)
            # min sorting algorithm for mlu is different from cpu,
            # when value is the same the min index may be different,
            # in this case, index test is not included for min in unit test.

    # @unittest.skip("not test")
    @testinfo()
    def test_min_other(self):
        type_list = [torch.float, torch.int]
        for t in type_list:
            for shape1,shape2 in [((1, 1, 1024), (64, 1024, 1)),
                                  ((2, 2, 4, 2), (2)),
                                  ((2, 2, 4, 2), (1, 2)),
                                  ((1, 2), (2, 2, 4 , 2)),
                                  ((2, 1, 2, 4), (1, 2, 4)),
                                  ((1, 2, 4), (2, 1, 2, 4)),
                                  ((1, 3, 1, 113, 1, 1, 1, 7), (13, 1, 17, 1, 31, 1, 1, 1)),
                                  ((255, 1, 5, 1, 1, 1, 1, 1), (1, 1, 1, 73, 1, 411, 1, 1)),
                                  ((257, 1, 1, 1, 1, 1, 1, 1), (1, 1, 13, 1, 1, 1, 1, 1)),
                                  ((), ()),
                                  ((), (1)),
                                  ((0), (0))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.min(x, y)
                out_mlu = torch.min(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_min(self):
        shape_list = [(2, 3, 4, 113, 4, 2, 1), (64, 3, 4),
                      (1, 32, 5, 12, 8), (2, 128, 10, 6),
                      (2, 512, 8), (1, 100), (24,),
                      (1, 1, 1, 73, 1, 411, 1, 1)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.min(x)
            out_mlu = torch.min(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_min_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.min(x)
        out_mlu = torch.min(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_min_out(self):
        for shape1,shape2 in [((1, 1, 1024), (64, 1024, 1)),
                              ((2, 1, 2, 4), (1, 2, 4)),
                              ((1, 2, 4), (2, 1, 2, 4)),
                              ((1, 3, 1, 113, 1, 1, 1, 7), (13, 1, 17, 1, 31, 1, 1, 1)),
                              ((255, 1, 5, 1, 1, 1, 1, 1), (1, 1, 1, 73, 1, 411, 1, 1)),
                              ((257, 1, 1, 1, 1, 1, 1, 1), (1, 1, 13, 1, 1, 1, 1, 1))]:
            x = torch.randn(shape1, dtype=torch.float)
            y = torch.randn(shape2, dtype=torch.float)
            out_cpu = torch.randn(1, dtype=torch.float)
            x_mlu = x.to('mlu')
            y_mlu = y.to('mlu')
            out_mlu = out_cpu.to('mlu')
            torch.min(x, y, out=out_cpu)
            torch.min(x_mlu, y_mlu, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_min_channels_last(self):
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[..., 0:1]]
        list_list = [func_list, func_list]
        for xin_func, yin_func in product(*list_list):
            for shape1, shape2 in [((2, 2, 2, 1, 2), (2, 2, 2, 2)),
                                   ((1, 3, 1, 224), (1, 3, 224, 224))]:
                x = self.convert_to_channel_last(torch.randn(shape1, dtype=torch.float))
                y = self.convert_to_channel_last(torch.randn(shape2, dtype=torch.float))
                x_mlu = x.to('mlu')
                y_mlu = y.to('mlu')
                out_cpu = torch.min(xin_func(x), yin_func(y))
                out_mlu = torch.min(xin_func(x_mlu), yin_func(y_mlu))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)


    # @unittest.skip("not test")
    @testinfo()
    def test_min_out_not_contiguous(self):
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[..., 0:1]]
        list_list = [func_list, func_list]
        for xin_func, yin_func in product(*list_list):
            for shape1, shape2 in [((2, 2, 2, 1, 2), (2, 2, 2, 2)),
                                   ((2, 1, 2, 4), (1, 2, 4)),
                                   ((1, 2, 4), (2, 1, 2, 4)),
                                   ((1, 3, 1, 1), (1, 3, 224, 224)),
                                   ((1, 3, 1, 224), (1, 3, 224, 224))]:
                x = torch.randn(shape1, dtype=torch.float)
                y = torch.randn(shape2, dtype=torch.float)
                x_mlu = x.to('mlu')
                y_mlu = y.to('mlu')
                out_cpu = torch.min(xin_func(x), yin_func(y))
                out_mlu = torch.min(xin_func(x_mlu), yin_func(y_mlu))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_exception(self):
        input = torch.randn((0)).to('mlu')
        ref_msg = "operation does not have an identity."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.min(input)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_other_exception(self):
        x = torch.randn(3, dtype=torch.half).to('mlu')
        y = torch.randn(3, dtype=torch.float).to('mlu')
        ref_msg = "Expected object of scalar type c10::Half but got scalar"\
                  " type float for argument 'other'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.min(x.to("mlu"), y.to("mlu"))
 
    # @unittest.skip("not test")
    @testinfo()
    def test_min_exception(self):
        input = torch.randn((0)).to('mlu')
        ref_msg = "operation does not have an identity."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.min(input)

if __name__ == '__main__':
    unittest.main()
