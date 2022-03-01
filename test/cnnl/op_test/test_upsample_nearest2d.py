from __future__ import print_function

import sys
import copy
import logging
import os
import itertools
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import math
import unittest
import torch
import torch.nn as nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411
logging.basicConfig(level=logging.DEBUG)

class TestUpsampleNearest2dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_nearest2d(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
        m = nn.UpsamplingNearest2d(scale_factor=2)
        type_list = [torch.float32]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[...,::2]]
        param_list = [shape_list, type_list, func_list]
        for shape, type, func in itertools.product(*param_list):
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
    def test_upsample_nearest2d_out(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype=type)
                out_cpu = torch.randn(1, dtype=type)
                out_mlu = torch.randn(1, dtype=type).to('mlu')
                output_size =  [int(math.floor(x.size(i + 2))) for i in range(2)]
                torch._C._nn.upsample_nearest2d(x, output_size, out=out_cpu) # pylint: disable=I1101, W0212
                torch._C._nn.upsample_nearest2d(x.to('mlu'), output_size, out=out_mlu)  # pylint: disable=I1101, W0212
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_nearest2d_exception(self):
        shape =(2, 3, 4, 5)
        m = nn.UpsamplingNearest2d(scale_factor=2)
        x = torch.randn(shape).to(torch.uint8)
        x_mlu = x.to("mlu")
        ref_msg = "Expected tensor for argument #1 'input' to have one of the"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu = m(x_mlu)

if __name__ == '__main__':
    unittest.main()
