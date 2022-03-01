from __future__ import print_function

import logging
import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = "OFF" # pylint: disable=C0413
import unittest
from itertools import product

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411
logging.basicConfig(level=logging.DEBUG)

class TestAsStrided(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_as_strided_orig(self):
        shape_list = [(2, 4), (2, 3, 4), (2, 2, 3, 4), (2, 2, 2, 2, 3)]
        size_list = [(1, 2), (2, 3), (1, 2, 3), (2, 2, 2)]
        stride_list = [(1, 2), (1, 1), (2, 1, 2), (1, 1, 1)]
        storage_offset_list = [1, 2]
        for item in product(shape_list, storage_offset_list):
            x = torch.randn(item[0], dtype=torch.float)
            x_mlu = x.to("mlu")
            for stride, size in zip(stride_list, size_list):
                out_cpu = torch.as_strided(x, size, stride, item[1])
                self.assertTrue(x.storage().data_ptr() == out_cpu.storage().data_ptr())
                out_mlu = torch.as_strided(x_mlu, size, stride, item[1])
                # (TODO) mlu tensor not support .storage() operator.
                self.assertTrue(x_mlu.data_ptr() == out_mlu.data_ptr() - 4 * item[1])
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

    #@unittest.skip("not test")
    @testinfo()
    def test_as_strided_channels_last(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)] 
        size_list = [(1, 2), (2, 3), (1, 2, 3), (2, 2, 2)] 
        stride_list = [(1, 2), (1, 1), (2, 1, 2), (1, 1, 1)] 
        storage_offset_list = [1, 2]
        for item in product(shape_list, storage_offset_list):
            x = torch.randn(item[0], dtype=torch.float).to(memory_format=torch.channels_last)
            x_mlu = x.to("mlu")
            for stride, size in zip(stride_list, size_list):
                out_cpu = torch.as_strided(x, size, stride, item[1])
                self.assertTrue(x.storage().data_ptr() == out_cpu.storage().data_ptr())
                out_mlu = torch.as_strided(x_mlu, size, stride, item[1])
                # (TODO) mlu tensor not support .storage() operator.
                self.assertTrue(x.stride() == x_mlu.stride())
                self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

    #@unittest.skip("not test")
    @testinfo()
    def test_as_strided_not_dense(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)] 
        size_list = [(1, 2), (2, 3), (1, 2, 3), (2, 2, 2)] 
        stride_list = [(1, 2), (1, 1), (2, 1, 2), (1, 1, 1)] 
        storage_offset_list = [1, 2]
        for item in product(shape_list, storage_offset_list):
            x = torch.randn(item[0], dtype=torch.float)
            x_cpu = x[:, :, :, 1:3]
            x_mlu = x.to("mlu")[:, :, :, 1:3]
            for stride, size in zip(stride_list, size_list):
                out_cpu = torch.as_strided(x_cpu, size, stride, item[1])
                self.assertTrue(x_cpu.storage().data_ptr() == out_cpu.storage().data_ptr())
                out_mlu = torch.as_strided(x_mlu, size, stride, item[1])
                # (TODO) mlu tensor not support .storage() operator.
                self.assertTrue(x_cpu.stride() == x_mlu.stride())
                self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())
                self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    #@unittest.skip("not test")
    @testinfo()
    def test_as_strided_exception(self):
        size_list = [(1,), (1, 2), (1, 2, 3), (1, 2), (2, 2)]
        stride_list = [(1, 2), (1,), (2, 3), (1, 2), (-1, 1)]
        storage_offset_list = [1, 0, 2, -1, 1]
        msg_list = [r"mismatch in length of strides and shape",
                    r"mismatch in length of strides and shape",
                    r"mismatch in length of strides and shape",
                    r"Tensor: invalid storage offset -1",
                    r"as_strided: Negative strides are not supported"]
        x = torch.randn((2, 3, 4), dtype=torch.float)
        x_mlu = x.to("mlu")
        for msg, stride, size, storage_offset in zip(msg_list, stride_list, size_list,
                                                     storage_offset_list):
            with self.assertRaisesRegex(RuntimeError, msg):
                out = torch.as_strided(x_mlu, size, stride, storage_offset)  # pylint: disable=W0612

if __name__ == '__main__':
    unittest.main()
