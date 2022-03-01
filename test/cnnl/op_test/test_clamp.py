from __future__ import print_function
import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")

from common_utils import testinfo, TestCase # pylint:disable=C0411,C0413

logging.basicConfig(level=logging.DEBUG)

class TestClampOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_clamp(self):
        shape_list = [((1, 1, 1, 2), (144, 7, 15, 2)),
                      ((5), (5)),
                      ((256, 144), (1))]
        dtype_list = [torch.float,
                      torch.half,
                      torch.int,
                      torch.int16,
                      torch.int8]  # half is not support for cpu
        min_list = (0.1, 1, None)
        max_list = (10, 100.1, None)
        product_list = product(shape_list,
                               dtype_list,
                               min_list,
                               max_list)
        for (shape1, shape2), dtype, min_, max_ in product_list:
            if max_ is None and min_ is None:
                continue
            if dtype == torch.half:
                x = torch.randn(shape1, dtype=torch.float)
                y = torch.randn(shape2, dtype=torch.float)
            else:
                x = torch.randn(shape1).to(dtype)
                y = torch.randn(shape2).to(dtype)
            out_cpu = torch.clamp(x, min_, max_)
            out_mlu = torch.clamp(self.to_mlu_dtype(x, dtype), min_, max_)
            if dtype is torch.half:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test clamp.out
            out_cpu = torch.clamp(x, min_, max_, out=y)
            out_mlu = torch.clamp(self.to_mlu_dtype(x, dtype),
                                  min_, max_,
                                  out=self.to_mlu_dtype(y, dtype))
            if dtype is torch.half:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test inplace operation
            x_cpu = copy.deepcopy(x)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x_cpu.clamp_(min_, max_)
            x_mlu.clamp_(min_, max_)
            if dtype is torch.half:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)


    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_backward(self):
        for shape in [(2,3), (8, 224, 224), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)

            out_cpu = torch.clamp(x, 0.1, 10)
            out_mlu = torch.clamp(self.to_device(x), 0.1, 10)

            grad = torch.randn(out_cpu.shape, dtype=torch.float)

            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)

            x_cpu = copy.deepcopy(x)
            x.grad.zero_()

            outmlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)

            self.assertEqual(outmlu_ptr, out_mlu.data_ptr(), 0)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)
            self.assertTensorsEqual(x, x_cpu, 0)

            # test inplace operation
            x.grad.zero_()

            x_mlu = self.to_device(x)
            x_mlu.clamp_(0.1, 10)
            xmlu_ptr = x_mlu.data_ptr()
            x_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)

            self.assertEqual(xmlu_ptr, x_mlu.data_ptr(), 0)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_not_dense(self):
        for shape in [(8, 224, 224, 16), (1, 3, 16, 16), (1, 3, 16, 14)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            input_cpu = x[:,:,:,3:6]
            out_cpu = torch.clamp(input_cpu, 0.1, 10)
            input_mlu = self.to_device(x)
            input_mlu = input_mlu[:,:,:,3:6]
            out_mlu = torch.clamp(input_mlu, 0.1, 10)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_channel_last(self):
        for shape in [(8, 224, 224, 16), (1, 3, 16, 16), (1, 3, 16, 14, 25)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            input_cpu = self.convert_to_channel_last(x)
            out_cpu = torch.clamp(input_cpu, 0.1, 10)
            input_mlu = self.to_device(x)
            input_mlu = self.convert_to_channel_last(input_mlu)
            out_mlu = torch.clamp(input_mlu, 0.1, 10)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

if __name__ == '__main__':
    unittest.main()
