from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestMaskedSelect(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,)]
        dtype = [torch.float, torch.half, torch.long, torch.int,
                 torch.int16, torch.int8, torch.bool]
        for type_ in dtype:
            for shape in shapes:
                x = torch.rand(shape).to(type_)
                x_mlu = self.to_device(x)
                if type_ == torch.half:
                    x = x.to(torch.float)
                mask = torch.randn(shape) > 0
                out_cpu = torch.masked_select(x, mask)

                mask_mlu = self.to_device(mask)
                out_mlu = torch.masked_select(x_mlu, mask_mlu)

                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_mask_select_broadcast(self):
        shapes = [((3,4,5),(4,5)), ((2,5,5,3),(3))]
        dtype = [torch.float, torch.half, torch.long, torch.int,
                 torch.int16, torch.int8, torch.bool]
        for type_ in dtype:
            for shape1, shape2 in shapes:
                a = torch.randn(shape1)
                b = torch.randn(shape2)
                a_mlu, b_mlu = a.to('mlu'), b.to('mlu')
                if type_ == torch.half:
                    a = a.to(torch.float)
                    b = b.to(torch.float)
                out_cpu = torch.masked_select(a, b > 0)
                out_mlu = torch.masked_select(a_mlu, b_mlu > 0)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.00)

                out_cpu = torch.masked_select(b, a > 0)
                out_mlu = torch.masked_select(b_mlu, a_mlu > 0)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select_out(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,)]
        out_shapes = [(512, 2, 5), (100, 512, 2), (512), (100, 1)]
        dtype = [torch.float, torch.half, torch.long, torch.int,
                 torch.int16, torch.int8, torch.bool]
        for type_ in dtype:
            for shape, out_shape in zip(shapes, out_shapes):
                x = torch.rand(shape).to(type_)
                x_mlu = self.to_device(x)
                mask = torch.randn(shape) > 0
                out_cpu = torch.rand(out_shape).to(type_)
                out_mlu = out_cpu.to(torch.device('mlu'))
                if type_ == torch.half:
                    x = x.to(torch.float)
                    out_cpu = out_cpu.to(torch.float)
                mask_mlu = self.to_device(mask)
                torch.masked_select(x, mask, out=out_cpu)
                torch.masked_select(x_mlu, mask_mlu, out=out_mlu)

                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select_channels_last_and_not_dense(self):
        # test channels last
        self_tensor = torch.randn(100,512,2,5)
        mask_tensor = torch.randn(100,512,2,5) > 0
        cpu_result = torch.masked_select(self_tensor.to(memory_format = torch.channels_last),
                                         mask_tensor.to(memory_format = torch.channels_last))

        device_result = torch.masked_select(
                self_tensor.to('mlu').to(memory_format = torch.channels_last),
                mask_tensor.to('mlu').to(memory_format = torch.channels_last))
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0.003, use_MSE = True)

        # test not dense
        self_tensor = self_tensor[...,:2]
        mask_tensor = mask_tensor[...,:2]
        cpu_result = torch.masked_select(self_tensor, mask_tensor)
        device_result = torch.masked_select(self_tensor.to('mlu'), mask_tensor.to('mlu'))
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0.003, use_MSE = True)

        # test channels last broadcast
        self_tensor = torch.randn(2, 5, 5, 3).to(memory_format = torch.channels_last)
        mask_tensor = torch.tensor([True, False, True])
        cpu_result = torch.masked_select(self_tensor, mask_tensor)
        device_result = torch.masked_select(self_tensor.to('mlu'), mask_tensor.to('mlu'))
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select_exception(self):
        self_tensor = torch.randn(1, 2, 5)
        mask_tensor = torch.randn(1, 2, 5).to(torch.uint8)
        ref_msg = "indexing with dtype torch.uint8 is now deprecated,"
        ref_msg = ref_msg + " please use a dtype torch.bool instead."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.masked_select(self_tensor.to('mlu'), mask_tensor.to('mlu'))

if __name__ == '__main__':
    unittest.main()
