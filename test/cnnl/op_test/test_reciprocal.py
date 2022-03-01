"""
test_reciprocal
"""
from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging
import copy

import torch
import torch_mlu.core.mlu_model as ct

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD+"/../../")

from common_utils import testinfo, TestCase # pylint: disable=C0413,E0401,C0411

logging.basicConfig(level=logging.DEBUG)

class TestReciprocalOp(TestCase):
    """
    test-reciprocal
    """
    #@unittest.skip("not test")
    @testinfo()
    def test_reciprocal(self):
        """
        test_reciprocal
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (1, 3, 2, 2), (3, 2, 4, 4), (3,20), (4)]:
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                    if isinstance(shape1, int) or \
                       (len(shape1) != 4 and memory_format == torch.channels_last):
                        continue
                    x_cpu = torch.rand(shape1, dtype=data_type)\
                            .to(memory_format = memory_format) + 0.00005
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type)

                    out_cpu = torch.reciprocal(x_cpu)
                    out_mlu = torch.reciprocal(x_mlu)

                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                            err, use_MSE=True)

                    x_cpu = torch.rand(shape1, dtype=data_type) + 0.00005
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type)

                    out_cpu = 1/x_cpu
                    out_mlu = 1/x_mlu

                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                            err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_reciprocal_inplace(self):
        """
        test_reciprocal_inplace
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (1, 3, 2, 2), (3, 2, 4, 4), (3,20), (4)]:
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                    if isinstance(shape1, int) or \
                       (len(shape1) != 4 and memory_format == torch.channels_last):
                        continue
                    x_cpu = torch.rand(shape1, dtype=data_type)\
                            .to(memory_format = memory_format) + 0.00005
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                    x_mlu_ptr = x_mlu.data_ptr()

                    x_cpu.reciprocal_()
                    x_mlu.reciprocal_()

                    self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(),
                                            err, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_reciprocal_out(self):
        """
        test_reciprocal_out
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (1, 3, 2, 2), (3, 2, 4, 4), (3,20), (4)]:
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                    if isinstance(shape1, int) or\
                       (len(shape1) != 4 and memory_format == torch.channels_last):
                        continue
                    x_cpu = torch.rand(shape1, dtype=data_type)\
                            .to(memory_format=memory_format) + 0.00005
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type) # pylint: disable=W0612
                    x_mlu_ptr = x_mlu.data_ptr()

                    out_cpu = torch.zeros(shape1, dtype=data_type)
                    out_mlu = torch.zeros(shape1, dtype=data_type).to("mlu")
                    torch.reciprocal(x_cpu, out=out_cpu)
                    torch.reciprocal(x_mlu, out=out_mlu)

                    self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                            err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float) + 0.00005
            x_mlu = x.to(ct.mlu_device())
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            out_cpu = x.reciprocal()
            out_mlu = self.to_mlu(x).reciprocal()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_inplace_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (10, 3, 32, 32), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float) + 0.00005
            x_mlu = copy.deepcopy(x).to(ct.mlu_device())
            if len(shape) == 4:
                x = x[:, :, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, :int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, :int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :int(shape[-1] / 2)]
            x_mlu = copy.deepcopy(x).to(ct.mlu_device())
            x.reciprocal_()
            x_mlu.reciprocal_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 3e-3, use_MSE=True)


if __name__ == '__main__':
    unittest.main()
