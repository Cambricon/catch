from __future__ import print_function
import logging
import sys
import os
import unittest

import torch
import torch_mlu.core.mlu_model as ct
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase # pylint: disable=C0413, C0411

def to_mlu(input_tensor):
    if input_tensor.device.type == 'mlu': # pylint: disable=R1705
        return input_tensor
    else:
        return input_tensor.to(ct.mlu_device())


class TestCatOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_cat_type(self):
        in_shape1 = (1, 2, 3)
        in_shape2 = (1, 77, 3)
        # TODO(CTR-3581):support uint8  # pylint: disable= W0511
        # dtypes = [torch.float, torch.int64, torch.long, torch.uint8]
        dtypes = [torch.float, torch.int64, torch.long, torch.half]
        for dtype in dtypes:
            if dtype is torch.half:
                input1 = torch.ones(in_shape1, dtype=torch.float)
                input2 = torch.ones(in_shape2, dtype=torch.float)
            else:
                input1 = torch.ones(in_shape1, dtype=dtype)
                input2 = torch.ones(in_shape2, dtype=dtype)

            inputs_cpu = [input1, input2]
            inputs_mlu = [self.to_mlu_dtype(input1, dtype), self.to_mlu_dtype(input2, dtype)]

            output_cpu = torch.cat(inputs_cpu, dim=1)
            output_mlu = torch.cat(inputs_mlu, dim=1)
            if dtype is torch.half:
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), 3e-3)
            else:
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_empty(self):
        in_shapes1 = [(0, 3, 32, 32), (4, 3, 32, 32)]
        in_shapes2 = [(0), (4, 32, 32)]
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for [in_shape1, in_shape2] in [in_shapes1, in_shapes2]:
            for dtype, err in dtypes:
                input1 = torch.randn(in_shape1, dtype=torch.float)
                input2 = torch.randn(in_shape2, dtype=torch.float)

                inputs_cpu = [input1, input2]
                inputs_mlu = [self.to_mlu_dtype(input1, dtype), self.to_mlu_dtype(input2, dtype)]

                output_cpu = torch.cat(inputs_cpu, dim=0)
                output_mlu = torch.cat(inputs_mlu, dim=0)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_channel_last(self):
        in_shapes1 = [(4, 5, 32, 32), (4, 3, 32, 32)]
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for [in_shape1, in_shape2] in [in_shapes1]:
            for dtype, err in dtypes:
                input1 = torch.randn(in_shape1, dtype=torch.float)
                input2 = torch.randn(in_shape2, dtype=torch.float)
                input_channels_last = self.convert_to_channel_last(input1)
                inputs_cpu = [input1, input2]
                inputs_mlu = [self.to_mlu_dtype(input_channels_last, dtype),
                              self.to_mlu_dtype(input2, dtype)]

                output_cpu = torch.cat(inputs_cpu, dim=1)
                output_mlu = torch.cat(inputs_mlu, dim=1)
                output_mlu_channels_first = output_mlu.cpu().float().contiguous()
                self.assertTensorsEqual(output_cpu, output_mlu_channels_first, err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_not_dense(self):
        in_shapes1 = [(4, 5, 32, 32), (4, 3, 32, 32)]
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for [in_shape1, in_shape2] in [in_shapes1]:
            for dtype, err in dtypes:
                input1 = torch.randn(in_shape1, dtype=torch.float)
                input2 = torch.randn(in_shape2, dtype=torch.float)
                inputs_cpu = [input1[:, :15], input2[:, :15]]
                inputs_mlu = [self.to_mlu_dtype(input1, dtype)[:, :15],
                              self.to_mlu_dtype(input2, dtype)[:, :15]]

                output_cpu = torch.cat(inputs_cpu, dim=1)
                output_mlu = torch.cat(inputs_mlu, dim=1)
                output_mlu_channels_first = output_mlu.cpu().float().contiguous()
                self.assertTensorsEqual(output_cpu, output_mlu_channels_first, err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_out(self):
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in dtypes:
            x = torch.randn((24, ), dtype=torch.float)
            x_mlu = self.to_mlu_dtype(x.clone(), data_type)

            out_cpu = torch.randn((4, ), dtype=torch.float)
            out_mlu = self.to_mlu_dtype(torch.randn((4, ), dtype=torch.float), data_type)

            torch.cat([x[:2], x[4:6]], out=out_cpu)
            torch.cat([x_mlu[:2], x_mlu[4:6]], out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_empty(self):
      a = torch.empty((0))
      b = torch.randn(4,3,32,32)
      cpu_result = torch.cat((a, b), dim = 1)
      mlu_result = torch.cat((a.to('mlu'), b.to('mlu')))
      self.assertTensorsEqual(cpu_result, mlu_result.cpu().float(), 0.0)


if __name__ == '__main__':
    unittest.main()
