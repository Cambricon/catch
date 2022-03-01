from __future__ import print_function

import sys
import copy
import logging
import os
import itertools
import math
import unittest
import torch
import torch.nn as nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411
logging.basicConfig(level=logging.DEBUG)

class TestUpsampleBilinear2dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_bilinear2d(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
        align_corners = [True, False]
        type_list = [torch.float32]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[...,::2]]
        param_list = [shape_list, align_corners, type_list, func_list]
        for shape, corner, type, func in itertools.product(*param_list):
            m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=corner)
            x = torch.randn(shape, dtype=type, requires_grad=True)
            out_cpu = m(func(x))
            out_mlu = m(func(self.to_mlu(x)))

            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_upsameple_bilinear2d_out(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
        align_corners = [True, False]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                for corner in align_corners:
                    x = torch.randn(shape, dtype=type)
                    x_mlu = x.to('mlu')
                    out_cpu = torch.randn(1, dtype=type)
                    out_mlu = torch.randn(1, dtype=type).to('mlu')
                    output_size =  [int(math.floor(x.size(i + 2))) for i in range(2)]
                    torch._C._nn.upsample_bilinear2d(x, output_size, corner, out=out_cpu)
                    torch._C._nn.upsample_bilinear2d(x_mlu, output_size, corner, out=out_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
