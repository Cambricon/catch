"""
test_div
"""
from __future__ import print_function

import unittest
import logging
import copy

import sys
import os
import itertools
import torch

os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF'
import torch_mlu.core.mlu_model as ct  # pylint: disable=C0413,W0611

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,E0401,C0411
from functools import wraps

logging.basicConfig(level=logging.DEBUG)


class TestTrueDivideOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_true_divide(self):
        type_list = [
            torch.bool, torch.int, torch.short, torch.int8, torch.uint8,
            torch.long, torch.float, torch.half
        ]
        for dtype in type_list:
            dividend = (torch.randn(5, device="mlu") * 100).to(dtype)
            divisor = torch.arange(1, 6, device="mlu").to(dtype)

            # Tests tensor / tensor division
            casting_result = dividend.to(
                torch.get_default_dtype()) / divisor.to(
                    torch.get_default_dtype())
            self.assertTensorsEqual(casting_result,
                                    torch.true_divide(dividend, divisor),
                                    4e-3,
                                    use_MSE=True)

            # Tests tensor/scalar division
            casting_result = dividend.to(torch.get_default_dtype()) / 2
            self.assertTensorsEqual(casting_result,
                                    torch.true_divide(dividend, 2.),
                                    4e-3,
                                    use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_true_divide_out(self):
        type_list = [
            torch.bool, torch.int, torch.short, torch.int8, torch.uint8,
            torch.long, torch.float, torch.half
        ]

        for dtype in type_list:
            dividend = (torch.randn(5, device="mlu") * 100).to(dtype)
            divisor = torch.arange(1, 6, device="mlu").to(dtype)

            # Tests that requests for an integer quotient fail
            if not dtype.is_floating_point:
                integral_quotient = torch.empty(5, device="mlu", dtype=dtype)
                with self.assertRaises(RuntimeError):
                    torch.true_divide(dividend, divisor, out=integral_quotient)
                with self.assertRaises(RuntimeError):
                    torch.true_divide(dividend, 2, out=integral_quotient)
            else:
                # Tests that requests for a floating quotient succeed
                floating_quotient = torch.empty(5, device="mlu", dtype=dtype)
                div_result = dividend / divisor
                self.assertTensorsEqual(div_result,
                                        torch.true_divide(
                                            dividend,
                                            divisor,
                                            out=floating_quotient),
                                        3e-3,
                                        use_MSE=True)
                self.assertTensorsEqual(
                    dividend / 2,
                    torch.true_divide(dividend, 2, out=floating_quotient),
                    3e-3,
                    use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_true_divide_inplace(self):
        type_list = [
            torch.bool, torch.int, torch.short, torch.int8, torch.uint8,
            torch.long, torch.float, torch.half
        ]
        for dtype in type_list:
            dividend = (torch.randn(5, device="mlu") * 100).to(dtype)
            divisor = torch.arange(1, 6, device="mlu").to(dtype)

        # Tests that requests for an integer quotient fail
        if not dtype.is_floating_point:
            with self.assertRaises(RuntimeError):
                dividend.true_divide_(divisor)
            with self.assertRaises(RuntimeError):
                dividend.true_divide_(2)
        else:
            # Tests that requests for a floating quotient succeed
            div_result = dividend.clone().div_(divisor)
            self.assertTensorsEqual(div_result,
                                    dividend.clone().true_divide_(divisor),
                                    3e-3,
                                    use_MSE=True)
            self.assertTensorsEqual(dividend.clone().div_(2),
                                    dividend.clone().true_divide_(2),
                                    3e-3,
                                    use_MSE=True)


if __name__ == '__main__':
    unittest.main()
