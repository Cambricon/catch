from __future__ import print_function
import logging
import sys
import os
import unittest

import torch
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
from common_utils import testinfo, TestCase # pylint: disable=C0413, C0411

torch.manual_seed(6503)

class TestCastOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_float_cast(self):
        shape = (2, 3, 4)
        type_list = [torch.half, torch.float,
                     torch.int, torch.short, torch.int8, torch.bool]
        for ori_t in type_list:
            x = torch.randn(shape, dtype=torch.float).to(ori_t)
            for tar_t in type_list:
                out_cpu = x.to(tar_t)
                out_mlu = self.to_mlu(x).to(tar_t)
                if tar_t is torch.half:
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True)
                else:
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_uint8_cast(self):
        shape = (2, 3, 4)
        #TODO(shangang): half and float cast to uint8 failed by cnnl op bug.
        type_list = [torch.int, torch.short, torch.int8, torch.bool]
        for ori_t in type_list:
            x = torch.randn(shape, dtype=torch.float).to(ori_t)
            t = torch.uint8
            out_cpu = x.to(t)
            out_mlu = self.to_mlu(x).to(t)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cast_channel_last_not_dense_last(self):
        shapes_list = [(64, 3, 7, 7),
                      (14, 7, 7, 7),
                      (3, 4, 5),
                      (3, 3, 4),
                      (5, 5, 5, 5)]
        for shape1 in shapes_list:
            input = torch.randn(shape1, dtype = torch.float)
            if input.dim() == 4:
                input = input.to(memory_format = torch.channels_last)
            input_mlu = input.to("mlu")

            # channels_last
            output_cpu = input.to(torch.int)
            output_mlu = input_mlu.to(torch.int)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.00, use_MSE=True)
            # not dense
            output_cpu_not_dense = input[:, :2].to(torch.int)
            output_mlu_not_dense = input_mlu[:, :2].to(torch.int)
            self.assertTensorsEqual(output_cpu_not_dense,
              output_mlu_not_dense.cpu(), 0.00, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
