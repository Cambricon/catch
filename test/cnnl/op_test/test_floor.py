from __future__ import print_function

import sys
import logging
import os
import copy
import unittest

import torch
import torch_mlu
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestFloorOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_floor(self):
        shape_list = [(1, 3, 224, 224), (2, 3, 4), (2, 2),
                      (254, 254, 112, 1, 1, 3), (0, 2, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.floor(x)
            out_mlu = torch.floor(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_floor_dtype(self):
        type_list = [torch.half,]
        for type_id in type_list:
            x = torch.randn((1, 3, 224, 224), dtype=torch.float)
            out_cpu = torch.floor(x.to(type_id).float())
            out_mlu = torch.floor(x.to(type_id).to('mlu'))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_floor_scalar(self):
        x_0 = torch.tensor(-1.57)
        out_cpu = torch.floor(x_0)
        out_mlu = torch.floor(self.to_mlu(x_0))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_floor_inplace(self):
        shape_list = [(1, 3, 224, 224), (2, 3, 4), (2, 2),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = self.to_mlu(x)
            y_data = y.data_ptr()
            torch.floor_(x)
            torch.floor_(y)
            self.assertEqual(y_data, y.data_ptr())
            self.assertTensorsEqual(x, y.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_floor_t(self):
        shape_list = [(1, 3, 224, 224), (2, 3, 4), (2, 2),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = x.floor()
            out_mlu = self.to_mlu(x).floor()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_floor_out(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (1000), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(1, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to(torch.device('mlu'))

            torch.floor(x, out=y)
            torch.floor(self.to_mlu(x), out=y_mlu)
            self.assertTensorsEqual(y, y_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_floor_channelslast_and_nodense(self):
        def run_test(x):
            out_cpu = torch.floor(x)
            out_mlu = torch.floor(x.to('mlu'))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(),
                                    0.0, use_MSE=True)

        shape_list = [(64, 3, 6, 6),
                      (2, 25, 64, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)

            # channels_last input
            run_test(x.to(memory_format = torch.channels_last))

            # not-dense input
            run_test(x[..., :2])

if __name__ == '__main__':
    unittest.main()
