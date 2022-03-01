from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import unittest
import logging

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_fill_(self):
        shape_list = [(10, 3, 512, 224), (2, 3, 4), (0, 3, 4), (2)]
        value_list = [2.3, 5, 0.59, -0.21]
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in dtype_list:
                input1 = torch.randn(shape_list[i], dtype=torch.float)
                out_cpu = torch.fill_(input1, value_list[i])
                out_mlu_1 = torch.fill_(self.to_mlu_dtype(input1, data_type), value_list[i])
                out_mlu_2 = torch.fill_(self.to_mlu_dtype(
                    input1, data_type), self.to_mlu(torch.tensor(value_list[i])))
                out_mlu_3 = self.to_mlu_dtype(input1, data_type).fill_(value_list[i])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_1.cpu().float(), err, use_MSE=True)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_2.cpu().float(), err, use_MSE=True)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_3.cpu().float(), err, use_MSE=True)
        for shape in shape_list:
            for data_type, err in dtype_list:
                tensor_1 = torch.randn(shape, dtype=torch.float)
                tensor_1_int = torch.tensor(2.55, dtype=torch.int32)
                tensor_2 = torch.randn(shape, dtype=torch.float)
                tensor_2_int = torch.tensor(3.55, dtype=torch.int32)
                tensor_1_copy = copy.deepcopy(tensor_1)
                tensor_2_copy = copy.deepcopy(tensor_2)
                out_cpu_1 = torch.fill_(tensor_1, tensor_1_int)
                out_cpu_2 = torch.fill_(tensor_2, tensor_2_int)
                out_mlu_1 = torch.fill_(self.to_mlu(tensor_1_copy), self.to_mlu(tensor_1_int))
                tensor_2_mlu = self.to_mlu_dtype(tensor_2_copy, data_type)
                tensor_2_mlu.fill_(self.to_mlu(tensor_2_int))
                self.assertTensorsEqual(
                    out_cpu_1.float(), out_mlu_1.cpu().float(), err, use_MSE=True)
                self.assertTensorsEqual(
                    out_cpu_2.float(), tensor_2_mlu.cpu().float(), err, use_MSE=True)

        for data_type, err in dtype_list:
            tensor_1 = torch.tensor(2.55, dtype=torch.float)
            tensor_1_int = torch.tensor(2.55, dtype=torch.int32)
            tensor_2 = torch.tensor(3.55, dtype=torch.float)
            tensor_2_int = torch.tensor(3.55, dtype=torch.int32)
            tensor_1_int_copy = copy.deepcopy(tensor_1_int)
            tensor_1_copy = copy.deepcopy(tensor_1)
            out_cpu_1 = torch.fill_(tensor_1_int, tensor_2)
            out_cpu_2 = torch.fill_(tensor_1, tensor_2_int)
            out_mlu_1 = torch.fill_(self.to_mlu(tensor_1_int_copy), self.to_mlu(tensor_2))
            out_mlu_2 = self.to_mlu_dtype(tensor_1_copy, data_type).fill_(self.to_mlu(tensor_2_int))
            self.assertTensorsEqual(
                out_cpu_1.float(), out_mlu_1.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(
                out_cpu_2.float(), out_mlu_2.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_fill_not_dense(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x[:, :, :, 1:3]
            x_mlu = x.to("mlu")[:, :, :, 1:3]
            out_cpu = x_cpu.fill_(2.33)
            out_mlu = x_mlu.fill_(2.33)
            self.assertTrue(x_cpu.storage().data_ptr() == out_cpu.storage().data_ptr())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTrue(x_mlu.data_ptr() == out_mlu.data_ptr())
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    #@unittest.skip("not test")
    @testinfo()
    def test_fill_channels_last(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x.to(memory_format=torch.channels_last)
            x_mlu = x.to("mlu").to(memory_format=torch.channels_last)
            out_cpu = x_cpu.fill_(2.33)
            out_mlu = x_mlu.fill_(2.33)
            self.assertTrue(x_cpu.storage().data_ptr() == out_cpu.storage().data_ptr())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTrue(x_mlu.data_ptr() == out_mlu.data_ptr())
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    #@unittest.skip("not test")
    @testinfo()
    def test_fill_exception(self):
        a = torch.randn((2, 3, 4)).to('mlu')
        value = torch.randn((2, 3, 4), dtype=torch.float).to('mlu')
        ref_msg = "fill_ only supports 0-dimension value tensor but got tensor with 3 dimensions."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.fill_(a, value)

if __name__ == "__main__":
    unittest.main()
