from __future__ import print_function

import sys
import os
import logging
import copy
import unittest

import torch
from torch import nn
import torch.nn.functional as F


import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

shape_list = [(), (2), (15),
              (1, 1), (45, 50),
              (1, 1, 1), (15, 224, 224),
              (1, 1, 1, 1), (1, 3, 224, 224)]
minmax_list = [(-0.2, 0.4), (-2, 2), (12, 24), (-24, -12)]
type_list = [torch.float, torch.half]
class TestHardtanhOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_contiguous(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)
                    output_cpu = hardtanh_layer(data)
                    output_mlu = hardtanh_layer(self.to_mlu(x))
                    self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
                    # test function:
                    output_cpu_f = F.hardtanh(data, min_v, max_v, inplace = False)
                    output_mlu_f = F.hardtanh(self.to_mlu(x), min_v, max_v, inplace = False)
                    self.assertTensorsEqual(output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_channel_last(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x = self.convert_to_channel_last(x)
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)
                    output_cpu = hardtanh_layer(data)
                    output_mlu = hardtanh_layer(self.to_mlu(x))
                    self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
                    # test function:
                    output_cpu_f = F.hardtanh(data, min_v, max_v, inplace = False)
                    output_mlu_f = F.hardtanh(self.to_mlu(x), min_v, max_v, inplace = False)
                    self.assertTensorsEqual(output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_not_dense(self):
        shape_list_not_dense = [(45, 100),
                                (15, 224, 448),
                                (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    if len(in_shape) == 4:
                        data = torch.randn(in_shape,
                                           dtype=torch.float)[:, :, :, :int(in_shape[-1] / 2)]
                    elif len(in_shape) == 3:
                        data = torch.randn(in_shape,
                                           dtype=torch.float)[:, :, :int(in_shape[-1] / 2)]
                    elif len(in_shape) == 2:
                        data = torch.randn(in_shape, dtype=torch.float)[:, :int(in_shape[-1] / 2)]
                    x = data.to(typeId)
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)
                    output_cpu = hardtanh_layer(data)
                    output_mlu = hardtanh_layer(self.to_mlu(x))
                    self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
                    # test function:
                    output_cpu_f = F.hardtanh(data, min_v, max_v, inplace = False)
                    output_mlu_f = F.hardtanh(self.to_mlu(x), min_v, max_v, inplace = False)
                    self.assertTensorsEqual(output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_inplace_contiguous(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x_cpu = copy.deepcopy(data)
                    x_mlu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x_mlu)
                    x_mlu_data_ptr = x_mlu.data_ptr()
                    F.hardtanh(x_mlu, min_v, max_v, inplace = True)
                    F.hardtanh(x_cpu, min_v, max_v, inplace = True)
                    self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

                    x_mlu_2 = copy.deepcopy(x)
                    x_mlu_2 = self.to_mlu(x_mlu_2)
                    x_mlu_2_data_ptr = x_mlu_2.data_ptr()
                    F.hardtanh_(x_mlu_2, min_v, max_v)
                    self.assertEqual(x_mlu_2_data_ptr, x_mlu_2.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu_2.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_inplace_channel_last(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x = self.convert_to_channel_last(x)
                    x_cpu = copy.deepcopy(data)
                    x_mlu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x_mlu)
                    x_mlu_data_ptr = x_mlu.data_ptr()
                    F.hardtanh(x_mlu, min_v, max_v, inplace = True)
                    F.hardtanh(x_cpu, min_v, max_v, inplace = True)
                    self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

                    x_mlu_2 = copy.deepcopy(x)
                    x_mlu_2 = self.to_mlu(x_mlu_2)
                    x_mlu_2_data_ptr = x_mlu_2.data_ptr()
                    F.hardtanh_(x_mlu_2, min_v, max_v)
                    self.assertEqual(x_mlu_2_data_ptr, x_mlu_2.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu_2.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_inplace_not_dense(self):
        shape_list_not_dense = [(45, 100),
                                (15, 224, 448),
                                (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    if len(in_shape) == 4:
                        data = torch.randn(in_shape,
                                           dtype=torch.float)[:, :, :, :int(in_shape[-1] / 2)]
                    elif len(in_shape) == 3:
                        data = torch.randn(in_shape,
                                           dtype=torch.float)[:, :, :int(in_shape[-1] / 2)]
                    elif len(in_shape) == 2:
                        data = torch.randn(in_shape,
                                           dtype=torch.float)[:, :int(in_shape[-1] / 2)]
                    x = data.to(typeId)
                    x_cpu = copy.deepcopy(data)
                    x_mlu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x_mlu)
                    x_mlu_data_ptr = x_mlu.data_ptr()
                    F.hardtanh(x_mlu, min_v, max_v, inplace = True)
                    F.hardtanh(x_cpu, min_v, max_v, inplace = True)
                    self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

                    x_mlu_2 = copy.deepcopy(x)
                    x_mlu_2 = self.to_mlu(x_mlu_2)
                    x_mlu_2_data_ptr = x_mlu_2.data_ptr()
                    F.hardtanh_(x_mlu_2, min_v, max_v)
                    self.assertEqual(x_mlu_2_data_ptr, x_mlu_2.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu_2.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_backward(self):
        for shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in [torch.float]:
                    #for typeId in type_list:
                    data = torch.randn(shape, dtype=torch.float, requires_grad=True)
                    x = data.to(typeId)
                    x_mlu = x.to(ct.mlu_device())
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)

                    out_cpu = hardtanh_layer(data)
                    out_mlu = hardtanh_layer(x_mlu)
                    grad = torch.randn(out_cpu.shape, dtype=torch.float)
                    grad_mlu = grad.to(ct.mlu_device())

                    out_cpu.backward(grad)
                    out_grad_cpu = copy.deepcopy(data.grad)
                    data.grad.zero_()
                    out_mlu.backward(grad_mlu)
                    out_grad_mlu = copy.deepcopy(data.grad)
                    if typeId == torch.float16:
                        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu().float(),
                                                0.02, use_MSE=True)
                    else:
                        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(),
                                                0.003, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
