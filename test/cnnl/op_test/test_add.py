# pylint: disable=W0223,R0201,C0413,C0411,C0301
from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH']='OFF' # pylint: disable=C0413
import copy
import unittest
import logging
import torch
from torch import nn
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase

class TestAddModel(nn.Module):
    def __init__(self):
        super(TestAddModel, self).__init__()

    def forward(self, x, y):
        z = x + y
        return z


class TestAddInplaceModel(nn.Module):
    def __init__(self):
        super(TestAddInplaceModel, self).__init__()

    def forward(self, x, y):
        x.add_(y)
        return x


class TestAddScaleModel(nn.Module):
    def __init__(self, scale):
        super(TestAddScaleModel, self).__init__()
        self.scale = scale

    def forward(self, x):
        y = x.add(self.scale)
        return y


class TestBroadCastAddModel(nn.Module):
    def __init__(self):
        super(TestBroadCastAddModel, self).__init__()

    def forward(self, x, y):
        z = x.add(y)
        return z


class TestAddOp(TestCase):      # pylint: disable=R0904
    #@unittest.skip("not test")
    @testinfo()
    def test_add(self):
        model = TestAddModel().float()
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape1, shape2 in [((10, 30, 50, 20), (10, 30, 50, 20))]:
            for data_type, err in dtype_list:
                input_self = torch.rand(shape1, dtype=torch.float)
                input_other = torch.rand(shape2, dtype=torch.float)
                out_cpu = model(input_self, input_other)
                out_mlu = model(self.to_mlu_dtype(input_self, data_type),
                                self.to_mlu_dtype(input_other, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

        for data_type, err in dtype_list:
            input_self = torch.tensor([0]).float()
            input_other = torch.tensor([[1, 0]]).float()
            out_cpu = model(input_self, input_other)
            out_mlu = model(self.to_mlu_dtype(input_self, data_type),
                            self.to_mlu_dtype(input_other, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            input_self = torch.tensor([0]).unsqueeze(1).float()
            input_other = torch.tensor([[1, 0]]).float()
            out_cpu = model(input_self, input_other)
            out_mlu = model(self.to_mlu_dtype(input_self, data_type),
                            self.to_mlu_dtype(input_other, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_inplace(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.int, 3e-3)]
        shape_list = [((1024, 12544), (1024, 12544)), ((1024, 1254), (1254)), \
                      ((10, 12, 10, 13), (10, 13)), ((10, 12, 10, 13), (12, 10, 13))]
        for data_type, err in dtype_list:
            for shape1, shape2 in shape_list:
                model_ = TestAddInplaceModel().float()
                input_self_cpu1 = (torch.rand(shape1, dtype=torch.float)*100).to(data_type)
                input_self_cpu2 = (torch.rand(shape2, dtype=torch.float)*100).to(data_type)
                input_self_mlu1 = self.to_mlu_dtype(copy.deepcopy(input_self_cpu1), data_type)
                input_self_mlu2 = self.to_mlu_dtype(copy.deepcopy(input_self_cpu2), data_type)
                input_ptr = input_self_mlu1.data_ptr()
                out_cpu = model_(input_self_cpu1, input_self_cpu2)
                out_mlu = model_(input_self_mlu1, input_self_mlu2)
                self.assertEqual(input_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_scale(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            model = TestAddScaleModel(0.5).float()
            input_self = torch.rand(1, 3, 224, 224, dtype=torch.float)
            out_cpu = model(input_self)
            out_mlu = model(self.to_mlu_dtype(input_self, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_scale_channel_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            model = TestAddScaleModel(0.5).float()
            input_self = torch.rand(1, 3, 224, 224).to(memory_format=torch.channels_last)
            out_cpu = model(input_self)
            out_mlu = model(self.to_mlu_dtype(input_self, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_scale_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            model = TestAddScaleModel(0.5).float()
            input_self = torch.rand(1, 3, 224, 224, dtype=torch.float)
            out_cpu = model(input_self[:, :, :, :112])
            out_mlu = model(self.to_mlu_dtype(input_self, data_type)[:, :, :, :112])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_broadcast_add(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            model_ = TestBroadCastAddModel().float()
            input_self_cpu = torch.ones(2, 1, 12, 1024, dtype=torch.float)
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), data_type)
            input_other = torch.rand(1, 1, 1, 1024, dtype=torch.float)
            out_cpu = model_(input_self_cpu, input_other)
            out_mlu = model_(input_self_mlu, self.to_mlu_dtype(input_other, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_tensor(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0.0)]
        for data_type, err in dtype_list:
            # [res] torch.add([res,] tensor1, tensor2)
            m1 = self.to_mlu_dtype(torch.randn(100, 100), data_type)
            v1 = self.to_mlu_dtype(torch.randn(100), data_type)

            ## contiguous
            res1 = torch.add(m1[4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(1)):
                res2[i] = m1[4, i] + v1[i]
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

            # non-contiguous
            res1 = torch.add(m1[:, 4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(0)):
                res2[i] = m1[i, 4] + v1[i]
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_inter_type(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # inter-type
            m1 = torch.randn(10, 10)
            res1 = m1 + 3
            res2 = self.to_mlu_dtype(m1, data_type) + torch.tensor(3).to(ct.mlu_device())
            self.assertTensorsEqual(res1.cpu(), res2.cpu().float(), err, use_MSE=True)
            res1 = 3 + m1
            res2 = torch.tensor(3).to(ct.mlu_device()) + self.to_mlu_dtype(m1, data_type)
            self.assertTensorsEqual(res1.cpu(), res2.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_empty(self):
        dtype_list = [torch.float, torch.half]
        for data_type in dtype_list:
            # 1d + empty
            m1 = self.to_mlu_dtype(torch.tensor([1.0], dtype=torch.float), data_type)
            m2 = self.to_mlu_dtype(torch.tensor([], dtype=torch.float), data_type)
            res = m1 + m2
            self.assertEqual(res.cpu().shape, m2.shape)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_bool(self):
        # bool
        m1 = torch.tensor([True, False, False, True, False, False],
                          dtype=torch.bool).to(ct.mlu_device())
        m2 = torch.tensor([True, True, False, False, False, True],
                          dtype=torch.bool).to(ct.mlu_device())
        expected = torch.tensor([True, True, False, True, False, True],
                                dtype=torch.bool)
        res = m1 + m2
        for val in range(expected.shape[0]):
            self.assertTrue((res.cpu()[val].item() == expected[val].item()))

    #@unittest.skip("not test")
    @testinfo()
    def test_add_multiply_add(self):
        # fused multiply add
        a = torch.zeros(10, dtype=torch.bool).to(ct.mlu_device())
        res = torch.add(a, a, alpha=0)
        expected = torch.zeros(10).bool()
        for val in range(expected.shape[0]):
            self.assertTrue((res.cpu()[val].item() == expected[val].item()))

    #@unittest.skip("not test")
    @testinfo()
    def test_add_dtype(self):
        model = TestAddModel()
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half, torch.double
        ]
        for input_t in type_list:
            for other_t in type_list:
                if input_t is torch.half:
                    input_self = torch.normal(
                        mean=0, std=torch.randn(20, dtype=torch.float))
                else:
                    input_self = torch.normal(
                        mean=0, std=torch.randn(20, dtype=torch.float)).to(input_t)

                if other_t is torch.half:
                    input_other = torch.normal(
                        mean=20.0, std=torch.randn(20, dtype=torch.float))
                else:
                    input_other = torch.normal(
                        mean=20.0, std=torch.randn(20, dtype=torch.float)).to(other_t)

                out_cpu = model(input_self, input_other)
                out_mlu = model(self.to_mlu_dtype(input_self, input_t),
                                self.to_mlu_dtype(input_other, other_t))
                if out_cpu.dtype == torch.bool:
                    for val in range(out_cpu[0]):
                        self.assertEqual(out_cpu[val].item(),
                                         out_mlu.cpu()[val].item())
                elif out_mlu.dtype == torch.half:
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                else:
                    self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_scalar(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape in [(224), (2, 4, 5, 3), (24, 24)]:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 + b.sum()
                out_mlu = 1.2 + self.to_mlu_dtype(b.sum(), data_type)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_batchnorm_add(self):
        b = torch.tensor(100, dtype=torch.long)
        out_cpu = b + 1
        out_mlu = b.to(ct.mlu_device()) + 1
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_inplace_intscalar(self):
        type_list = [
            torch.float, torch.int, torch.short, torch.int8, torch.uint8,
            torch.long, torch.half, torch.double
        ]
        for input_t in type_list:
            if input_t is torch.half:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float))
            else:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float)).to(input_t)
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_ptr = input_self_mlu.data_ptr()
            input_self_cpu += 1
            input_self_mlu += 1
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_inplace_boolscalar(self):
        input_cpu = torch.randint(100, (3, 5, 7, 9))
        input_mlu = input_cpu.to("mlu")
        input_mlu_ptr = input_mlu.data_ptr()
        input_cpu.add_(True)
        input_mlu.add_(True)
        self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
        self.assertTensorsEqual(input_cpu.float(), input_mlu.cpu().float(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_inplace_floatscalar(self):
        type_list = [torch.float, torch.half]
        for input_t in type_list:
            if input_t is torch.half:
                input_self_cpu = torch.normal(mean=20,
                                              std=torch.randn(20, dtype=torch.float))
            else:
                input_self_cpu = torch.normal(mean=20,
                                              std=torch.randn(20, dtype=torch.float)).to(input_t)
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_ptr = input_self_mlu.data_ptr()
            input_self_cpu += 2.3
            input_self_mlu += 2.3
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_scalar_dtype(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half, torch.double
        ]
        for scalar in [3, 3.3, True]:
            for type_t in type_list:
                if type_t is torch.half:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float))
                else:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float)).to(type_t)
                input_self_mlu = self.to_mlu_dtype(input_self_cpu, type_t)
                out_cpu = input_self_cpu + scalar
                out_mlu = input_self_mlu + scalar
                if out_cpu.dtype == torch.bool:
                    self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                    for val in range(out_cpu[0]):
                        self.assertEqual(out_cpu[val].item(),
                                         out_mlu.cpu()[val].item())
                elif out_mlu.dtype == torch.half:
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                else:
                    self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_scalar_add_dtype(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half, torch.double
        ]
        for scalar in [3, 3.3, True]:
            for type_t in type_list:
                if type_t is torch.half:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float))
                else:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float)).to(type_t)
                input_self_mlu = self.to_mlu_dtype(input_self_cpu, type_t)
                out_cpu = scalar + input_self_cpu
                out_mlu = scalar + input_self_mlu
                if out_cpu.dtype == torch.bool:
                    self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                    for val in range(out_cpu[0]):
                        self.assertEqual(out_cpu[val].item(),
                                         out_mlu.cpu()[val].item())
                elif out_mlu.dtype == torch.half:
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                else:
                    self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_value(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0.0)]
        for data_type, err in dtype_list:
            # [res] torch.add([res,] tensor, value)
            m1 = self.to_mlu_dtype(torch.randn(10, 10), data_type)

            # contiguous
            res1 = m1.clone()
            res1[3].add_(2)
            res2 = m1.clone()
            for i in range(m1.size(1)):
                res2[3, i] = res2[3, i] + 2
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

            # non-contiguous
            m1 = self.to_mlu_dtype(torch.randn(10, 10), data_type)
            res1 = m1.clone()
            res1[:, 3].add_(2)
            res2 = m1.clone()
            for i in range(m1.size(0)):
                res2[i, 3] = res2[i, 3] + 2
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_channels_last(self):
        shapes_list = [((64, 3, 7, 7), (7, 7)),
                      ((14, 7, 7, 7), (7)),
                      ((3, 4, 5), (2, 3, 4, 5)),
                      ((3, 3, 3), (3, 3, 3, 3)),
                      ((5, 5, 5, 5), (5, 5, 5, 5))]
        for shape1, shape2 in shapes_list:
            input = torch.randn(shape1, dtype = torch.float)
            other = torch.randn(shape2, dtype = torch.float)
            if input.dim() == 4:
                input = input.to(memory_format = torch.channels_last)
            if other.dim() == 4:
                other = other.to(memory_format = torch.channels_last)
            input_mlu = input.to("mlu")
            other_mlu = other.to("mlu")

            # channels_last
            output_cpu = input + other
            output_mlu = input_mlu + other_mlu
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.00, use_MSE=True)

            # channels_last and inplace
            if input.dim() >= other.dim():
                input.add_(other)
                input_mlu_ptr = input_mlu.data_ptr()
                input_mlu.add_(other_mlu)
                self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)
                self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_add_not_dense(self):
        shapes_list = [((64, 3, 7, 7), (7, 7))]
        for shape1, shape2 in shapes_list:
            input = torch.randn(shape1, dtype = torch.float)
            other = torch.randn(shape2, dtype = torch.float)
            input_mlu = input.to("mlu")
            other_mlu = other.to("mlu")

            output_cpu = input[:, :, :, :5] + other[:, :5]
            output_mlu = input_mlu[:, :, :, :5] + other_mlu[:, :5]
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.00, use_MSE=True)

            if input.dim() >= other.dim():
                input[:, :, :, :5].add_(other[:, :5])
                input_mlu_ptr = input_mlu.data_ptr()
                input_mlu[:, :, :, :5].add_(other_mlu[:, :5])
                self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)
                self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_add_exception(self):
        a = torch.randn(3).to('mlu')
        b = torch.randn(3).to('mlu')
        ref_msg = "Boolean alpha only supported for Boolean results"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.add(a, b, alpha=True)

        a = torch.randn(3).int().to('mlu')
        b = torch.randn(3).int().to('mlu')
        ref_msg = "For integral input tensors, argument alpha must not be a floating point number."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.add(a, b, alpha=2.1)


if __name__ == '__main__':
    unittest.main()
