from __future__ import print_function

import sys
import os
import logging
import unittest

import torch
import torch.nn as nn
torch.set_grad_enabled(False)

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")

from common_utils import TestCase  # pylint: disable=C0413, C0411
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestReshapeModel(nn.Module):
    def __init__(self):
        super(TestReshapeModel, self).__init__()

    def forward(self, x):
        out = x.reshape(1, -1)
        return out

class TestReshapeOp(TestCase):
    # @unittest.skip("not test")
    def test_reshape(self):
        # init: None | input: tensor
        gen_types = '|t'
        gen_params = [((1,32,10),),
                      ((1,4,32,32),),
                      ((1,58,2,28,28),),
                      ((1,3,2,12,28,6),)]
        self.running_mode = "fusion"
        cases = self.gen_by_params(gen_types, gen_params)
        self._test_several_cases(cases,
                                 TestReshapeModel,
                                 0.003,
                                 use_MSE=True)

    # @unittest.skip("not test")
    def test_reshape_types(self):
        for in_shape, out_shape in [((2,1),(2)),((1, 1000, 1, 1),(1,-1)),
                                    ((1,3,200,200),(1,-1,1,200,200)),
                                    ((1,4,32,32), (1,-1)),
                                    ((1,3,10,8,12,6), (1,-1,12,6))]:
            for element_type in [torch.half, torch.float, torch.int, torch.short, \
                                 torch.long, torch.uint8, torch.int8, torch.bool]:
                int_tensor = torch.randint(3, 5, in_shape)
                if element_type == torch.half:
                    x_cpu = (int_tensor + torch.randn(in_shape, dtype=torch.float)).half().float()
                else:
                    x_cpu = (int_tensor+torch.randn(in_shape)).to(dtype=element_type)
                x_mlu = self.to_mlu(x_cpu).to(dtype=element_type)
                y_cpu = x_cpu.reshape(out_shape)
                y_mlu = x_mlu.reshape(out_shape)
                if element_type == torch.half:
                    self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), 0.003, use_MSE = True)
                else:
                    assert(torch.all(y_cpu.eq(y_mlu.cpu())))

if __name__ == '__main__':
    unittest.main()
