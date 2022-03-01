from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(2)

class TestLinspaceOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_linspace(self):
        start_list = [1, 3, 3.5, 3.5, 4.1, 8.9, 11]
        end_list = [2, 5, 2.5, 10.5, 11.3, 99.1, 121]
        steps_list = [0, 3, 1, 11, 6, 100, 121]
        type_list = [(torch.float, torch.half), (torch.float, torch.float)]
        err = 1e-7
        for t1, t2 in type_list: # pylint: disable=C0200
            for start, end, steps in product(start_list, end_list, steps_list):  # pylint: disable=C0200
                # default support fp32 and fp16if t == torch.half:
                if t2 == torch.half:
                    err = 1e-1

                x = torch.linspace(start, end, steps=steps, device="cpu", dtype=t1)
                x_mlu = torch.linspace(start, end, steps=steps, device="mlu", dtype=t2)
                self.assertTensorsEqual(x, x_mlu.cpu(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_linspace_out(self):
        start_list = [1, 3, 3.5, 3.5, 4.1, 8.9, 11]
        end_list = [2, 5, 2.5, 10.5, 11.3, 99.1, 121]
        steps_list = [0, 3, 1, 1, 11, 6, 121]
        type_l = [torch.float, torch.half]
        err = 1e-7
        for t in type_l:
            for start, end, steps in product(start_list, end_list, steps_list):  # pylint: disable=C0200
                in1 = torch.randn(1).to(t)
                in1_mlu = in1.to('mlu')
                if t == torch.half:
                    in1 = torch.randn(1)
                    err = 1e-1

                x = torch.linspace(start, end, steps=steps,
                                        device="cpu", out=in1)
                x_mlu = torch.linspace(start, end, steps=steps,
                                        out=in1_mlu)
                self.assertTensorsEqual(x, x_mlu.cpu(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_linspace_exception(self):
        ref_msg = "number of steps must be non-negative"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.linspace(1, 10, -1, device='mlu')

if __name__ == '__main__':
    unittest.main()
