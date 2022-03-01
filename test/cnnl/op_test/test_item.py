from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413
import torch_mlu.core.mlu_model as ct

logging.basicConfig(level=logging.DEBUG)

class TestItem(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_item(self):
        shape_list = ((),(1))
        type_list = [torch.float, torch.half, torch.int, torch.short,
            torch.long, torch.int8, torch.uint8, torch.bool]
        for t in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype=torch.float)
                x = x.to(t)
                out_cpu = x.item()
                out_mlu = x.to(ct.mlu_device()).item()
                self.assertEqual(out_cpu, out_mlu)
                self.assertEqual(True, type(out_cpu) == type(out_mlu))

    #@unittest.skip("not test")
    @testinfo()
    def test_item_exception(self):
        x = torch.randn([1,2], dtype=torch.float)
        x = x.to("mlu")
        ref_msg = "only one element tensors can be converted to Python scalars"
        with self.assertRaisesRegex(ValueError, ref_msg):
            x.item()

if __name__ == '__main__':
    unittest.main()
