from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
import torch.nn.functional as F
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestGeluOp(TestCase):

    #@unittest.skip("not test")
    @testinfo()
    def test_gelu_boundary_value(self):
        for number in [0, 0.0001, -0.0001, 999999999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.gelu(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.gelu(self.to_mlu(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003,
                use_MSE=True)
            self.assertTensorsEqual(input_cpu, x, 0.003,
                use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gelu(self):
        shape_list = [(50), (35, 46), (16, 27, 38), (128, 4, 128, 124), (14, 19, 11, 13, 21),
                    (6, 7, 8, 9, 10, 11), (16, 17, 18, 19, 20, 21)]
        type_list = [torch.float, torch.half]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x:x]
        list_list = [type_list, shape_list, mode_list]

        for dtype, shape, mode in product(*list_list):
            x_0 = torch.randn(shape, dtype=torch.float, requires_grad=False)
            x = x_0.to(dtype)
            x_mlu = x.to(ct.mlu_device())

            x_0 = mode(x_0)
            x_0.requires_grad = True
            x_mlu = mode(x_mlu)
            x_mlu.requires_grad = True

            out_cpu = F.gelu(x_0)
            out_mlu = F.gelu(x_mlu)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.to(ct.mlu_device())

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x_0.grad)
            x_0.grad.zero_()
            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x_mlu.grad)

            self.assertTensorsEqual(
                out_cpu,
                out_mlu.cpu().float(),
                0.03 if dtype == torch.half else 0.003,
                use_MSE=True)

            self.assertTensorsEqual(
                out_grad_cpu,
                out_grad_mlu.cpu().float(),
                0.03 if dtype == torch.half else 0.003,
                use_MSE=True)

if __name__ == '__main__':
    unittest.main()
