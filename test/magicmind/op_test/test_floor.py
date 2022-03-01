from __future__ import print_function
import torch
import torch.nn as nn
import torch_mlu
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import sys
import os
import copy

import unittest
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

class TestFloorModel(nn.Module):
    def __init__(self):
        super(TestFloorModel, self).__init__()

    def forward(self, x):
        y = torch.floor(x)
        return y

class TestFloorInplaceModel(nn.Module):
    def __init__(self):
        super(TestFloorInplaceModel, self).__init__()

    def forward(self, x):
        torch.floor_(x)
        return x

class TestFloorOp(TestCase):
    # @unittest.skip("not test")
    def test_floor(self):
        gen_types = '|t'
        gen_params = [((2),),
                      ((2,24),),
                      ((2,24,56),),
                      ((2,24,56,28),),
                      ((2,24,56,28,10),)]
        for data_type in [torch.FloatTensor, torch.HalfTensor]:
            self.data_type = data_type
            cases = self.gen_by_params(gen_types, gen_params)
            self.running_mode = "fusion"
            self._test_several_cases(cases,
                                    TestFloorModel,
                                    0.003,
                                    use_MSE=True)

            self._test_several_cases(cases,
                                    TestFloorInplaceModel,
                                    0.003,
                                    use_MSE=True)

if __name__ == '__main__':
    unittest.main()
