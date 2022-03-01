from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestMaskedFill(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_tensor(self):
        types = [torch.half, torch.float, torch.double]
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100, ), ()]
        err = 0.0
        for t, shape in product(types, shapes):
            x = torch.rand(shape, dtype=t)
            mask = torch.ones(shape, dtype=torch.bool)
            value = torch.tensor(2.33, dtype=t)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            value_mlu = self.to_device(value)
            ori_ptr = x_mlu.data_ptr()
            if t == torch.half:
                x, value = x.float(), value.float()
                err = 0.003
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(
                x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_channels_last_and_not_dense(self):
            shape = (100, 512, 2, 5)
            # channels last
            x = torch.rand(shape, dtype=torch.float)
            mask = torch.ones(shape, dtype=torch.bool)
            value = torch.tensor(2.33, dtype=torch.float)
            value_mlu = self.to_device(value)
            x = x.to(memory_format = torch.channels_last)
            mask = mask.to(memory_format = torch.channels_last)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003,
                                    use_MSE = True)
            # not dense
            x = torch.rand(shape, dtype=torch.float)[...,:2]
            mask = torch.ones(shape, dtype=torch.bool)[...,:2]
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003,
                                    use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_scalar(self):
        types = [torch.half, torch.float, torch.double]
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100, ), ()]
        err = 0.0
        for t, shape in product(types, shapes):
            x = torch.rand(shape, dtype=t)
            mask = torch.ones(shape, dtype=torch.bool)
            value = 3.14159
            x_mlu = self.to_device(x)
            ori_ptr = x_mlu.data_ptr()
            mask_mlu = self.to_device(mask)
            if t == torch.half:
                x = x.float()
                err = 0.003
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value)
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(
                x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())

if __name__ == '__main__':
    unittest.main()
