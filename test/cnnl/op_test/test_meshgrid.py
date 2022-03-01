from __future__ import print_function

from itertools import product
import sys
import os
import logging
import unittest
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import TestCase, testinfo  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_meshgrid(self):
        shape_lst1 = [(1,), (3,), (7,), (111,)]
        shape_lst2 = [(0,), (2,), (8,), (256,)]
        shape_lst3 = [(5,), (15,), (125,)]
        dtype_lst = [torch.float, torch.int64, torch.long]
        loop_val = [shape_lst1, shape_lst2, shape_lst3, dtype_lst]
        for param in product(*loop_val):
            shape1, shape2, shape3, dtype = param
            if dtype==torch.float:
                input1 = torch.randn(shape1, dtype=dtype)
                input2 = torch.randn(shape2, dtype=dtype)
                input3 = torch.randn(shape3, dtype=dtype)
            else:
                input1 = torch.randint(128, shape1, dtype=dtype)
                input2 = torch.randint(128, shape2, dtype=dtype)
                input3 = torch.randint(128, shape3, dtype=dtype)
            input1_mlu = input1.to(torch.device("mlu"))
            input2_mlu = input2.to(torch.device("mlu"))
            input3_mlu = input3.to(torch.device("mlu"))
            output = torch.meshgrid(input1, input2, input3)
            output_mlu = torch.meshgrid(input1_mlu, input2_mlu, input3_mlu)
            for i in range(3):
                self.assertTensorsEqual(output[i], output_mlu[i].cpu(), 0)

if __name__ == "__main__":
    unittest.main()
