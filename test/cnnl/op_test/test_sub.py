from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import torch
from torch import nn
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF'

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413
import torch_mlu.core.mlu_model as ct  # pylint: disable=C0413

class TestSubModel(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(TestSubModel, self).__init__()

    def forward(self, x, y):  # pylint: disable=R0201
        z = x - y
        return z


class TestSubInplaceModel(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(TestSubInplaceModel, self).__init__()

    def forward(self, x, y):  # pylint: disable=R0201
        x.sub_(y)
        return x


class TestSubScaleModel(nn.Module):  # pylint: disable=W0223
    def __init__(self, scale):
        super(TestSubScaleModel, self).__init__()
        self.scale = scale

    def forward(self, x):
        y = x.sub(self.scale)
        return y


class TestSubScalealphaModel(nn.Module):  # pylint: disable=W0223
    def __init__(self, scale, alpha):
        super(TestSubScalealphaModel, self).__init__()
        self.scale = scale
        self.alpha = alpha

    def forward(self, x):
        y = x.sub(self.scale, self.alpha)
        return y

class TestBroadCastSubModel(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(TestBroadCastSubModel, self).__init__()

    def forward(self, x, y):  # pylint: disable=R0201
        z = x.sub(y)
        return z


class TestSubOp(TestCase):          # pylint: disable=R0904
    # @unittest.skip("not test")
    @testinfo()
    def test_sub(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model = TestSubModel().float()
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

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_channel_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model = TestSubModel().float()
        for shape1, shape2 in [((10, 30, 50, 20), (10, 30, 50, 20))]:
            for data_type, err in dtype_list:
                input_self = torch.rand(shape1).to(memory_format=torch.channels_last)
                input_other = torch.rand(shape2).to(memory_format=torch.channels_last)
                out_cpu = model(input_self, input_other)
                out_mlu = model(self.to_mlu_dtype(input_self, data_type),
                                self.to_mlu_dtype(input_other, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model = TestSubModel().float()
        for shape1, shape2 in [((10, 30, 50, 20), (10, 30, 50, 20))]:
            for data_type, err in dtype_list:
                input_self = torch.rand(shape1)
                input_other = torch.rand(shape2)
                out_cpu = model(input_self[::2], input_other[::2])
                out_mlu = model(self.to_mlu_dtype(input_self, data_type)[::2],
                                self.to_mlu_dtype(input_other, data_type)[::2])
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model_ = TestSubInplaceModel().float()
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1)),
                                   ((2, 30, 80), (2, 30, 80)),
                                   ((3, 20), (3, 20)),
                                   ((10), (10)),
                                   ((2, 1, 2, 4), (1, 2, 4)),
                                   #((1, 3, 224, 224), (1, 1, 1, 1)),
                                   #((1, 3, 224, 224), (1)),  maybe bigger than 3e-3 if the value is round of 0. # pylint: disable=C0301
                                   ((1, 3, 224), (1, 3, 1))]:
                input_self_cpu = torch.ones(shape1, dtype=torch.float)
                input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), data_type)
                input_ptr = input_self_mlu.data_ptr()
                input_other = torch.rand(shape2, dtype=torch.float)
                out_cpu = model_(input_self_cpu, input_other)
                out_mlu = model_(input_self_mlu, self.to_mlu_dtype(input_other, data_type))
                self.assertEqual(out_mlu.data_ptr(), input_ptr)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace_channel_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model_ = TestSubInplaceModel().float()
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1))]:
                input_self_cpu = torch.ones(shape1).to(memory_format=torch.channels_last)
                input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), data_type)
                input_ptr = input_self_mlu.data_ptr()
                input_other = torch.rand(shape2).to(memory_format=torch.channels_last)
                out_cpu = model_(input_self_cpu, input_other)
                out_mlu = model_(input_self_mlu, self.to_mlu_dtype(input_other, data_type))
                self.assertEqual(out_mlu.data_ptr(), input_ptr)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model_ = TestSubInplaceModel().float()
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1))]:
                input_self_cpu = torch.ones(shape1)
                input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), data_type)
                input_ptr = input_self_mlu.data_ptr()
                input_other = torch.rand(shape2)
                input_other_mlu = self.to_mlu_dtype(input_other, data_type)
                out_cpu = model_(input_self_cpu[:, :, :112, :], input_other[:, :, :112, :])
                out_mlu = model_(input_self_mlu[:, :, :112, :],
                                input_other_mlu[:, :, :112, :])
                self.assertEqual(out_mlu.data_ptr(), input_ptr)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_scale(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model_list = [TestSubScaleModel(0.5).float(), TestSubScalealphaModel(0.5, 1.5).float(),]
        for data_type, err in dtype_list:
            for model in model_list:
                input_self = torch.rand(1, 3, 224, 224, dtype=torch.float)
                out_cpu = model(input_self)
                out_mlu = model(self.to_mlu_dtype(input_self, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_scale_channel_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model = TestSubScaleModel(0.5).float()
        for data_type, err in dtype_list:
            input_self = torch.rand(1, 3, 224, 224).to(memory_format=torch.channels_last)
            out_cpu = model(input_self)
            out_mlu = model(self.to_mlu_dtype(input_self, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_scale_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model = TestSubScaleModel(0.5).float()
        for data_type, err in dtype_list:
            input_self = torch.rand(1, 3, 224, 224)
            input_mlu = self.to_mlu_dtype(input_self, data_type)
            out_cpu = model(input_self[:, :, :, :112])
            out_mlu = model(input_mlu[:, :, :, :112])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_broadcast_sub(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        model_ = TestBroadCastSubModel().float()
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1)),
                                   ((2, 30, 80), (2, 30, 80)),
                                   ((3, 20), (3, 20)),
                                   ((10), (10)),
                                   ((2, 1, 2, 4), (1, 2, 4)),
                                   #((1, 3, 224, 224), (1, 1, 1, 1)),
                                   #((1, 3, 224, 224), (1)),  maybe bigger than 3e-3 if the value is round of 0. # pylint: disable=C0301
                                   ((1, 3, 224), (1, 3, 1))]:
                input_self_cpu = torch.ones(shape1, dtype=torch.float)
                input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), data_type)
                input_other = torch.rand(shape2, dtype=torch.float)
                out_cpu = model_(input_self_cpu, input_other)
                out_mlu = model_(input_self_mlu, self.to_mlu_dtype(input_other, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_tensor(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0.0)]
        for data_type, err in dtype_list:
            # [res] torch.sub([res,] tensor1, tensor2)
            m1 = torch.randn(100, 100).to(ct.mlu_device()).to(data_type)
            v1 = torch.randn(100).to(ct.mlu_device()).to(data_type)

            # contiguous
            res1 = torch.sub(m1[4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(1)):
                res2[i] = m1[4, i] - v1[i]
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

            # non-contiguous
            res1 = torch.sub(m1[:, 4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(0)):
                res2[i] = m1[i, 4] - v1[i]
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    # TODO(zhangguopeng): master not share storage feature, so res1[3] return an tmporary tensor, pylint: disable=W0511
    # res1 and res2 are still unchanged.
    @testinfo()
    def test_sub_value(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # [res] torch.sub([res,] tensor, value)
            m1 = torch.randn(10, 10).to(ct.mlu_device()).to(data_type)

            # contiguous
            res1 = m1.clone()
            res1[3].sub_(2)
            res2 = m1.clone()
            for i in range(m1.size(1)):
                res2[3, i] = res2[3, i] - 2
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

            # non-contiguous
            m1 = torch.randn(10, 10)
            m2 = m1.to(ct.mlu_device()).to(data_type)
            res1 = m2.clone()
            res1_data_ptr1 = res1.data_ptr()
            res1[:, 3].sub_(2)
            res1_data_ptr2 = res1.data_ptr()
            res2 = m2.clone()
            res2_data_ptr1 = res2.data_ptr()
            for i in range(m1.size(0)):
                res2[i, 3] = res2[i, 3] - 2
            res2_data_ptr2 = res2.data_ptr()
            cpu_input = m1.clone()
            cpu_input[:, 3].sub_(2)
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(cpu_input, res2.cpu().float(), err, use_MSE=True)
            self.assertEqual(res1_data_ptr1, res1_data_ptr2)
            self.assertEqual(res2_data_ptr1, res2_data_ptr2)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inter_type(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # inter-type
            m1 = torch.randn(10, 10).to(ct.mlu_device()).to(data_type)
            res1 = m1 - 3
            res2 = m1 - torch.tensor(3).to(ct.mlu_device())
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)
            res1 = 3 - m1
            res2 = torch.tensor(3).to(ct.mlu_device()) - m1
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_empty(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [(), (1)]
        for data_type in dtype_list:
            # (0d or 1d) - empty
            for shape in shape_list:
                m1 = torch.rand(shape, dtype=torch.float).to(ct.mlu_device()).to(data_type)
                m2 = torch.tensor([], dtype=torch.float).to(ct.mlu_device()).to(data_type)
                res = m1 - m2
                self.assertEqual(res.cpu().shape, m2.shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_dtype(self):
        model = TestSubModel()
        type_list = [
            torch.float, torch.int, torch.short, torch.int8,
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
                if out_mlu.dtype == torch.half:
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

    # @unittest.skip("not test")
    @testinfo()
    def test_scalar(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape in [(224), (2, 4, 5, 3), (24, 24)]:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 - b.sum()
                out_mlu = 1.2 - self.to_mlu_dtype(b.sum(), data_type)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
                c = torch.rand(shape, dtype=torch.float)
                out_cpu = c.sum() - 1.2
                out_mlu = self.to_mlu_dtype(c.sum(), data_type) - 1.2
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_batchnorm_sub(self):
        b = torch.tensor(100, dtype=torch.long)
        out_cpu = b - 1
        out_mlu = b.to(ct.mlu_device()) - 1
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)
        c = torch.tensor(100, dtype=torch.long)
        out_cpu = 1 - c
        out_mlu = 1 - c.to(ct.mlu_device())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace_intscalar(self):
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
            input_self_cpu -= 1
            input_self_mlu -= 1
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace_floatscalar(self):
        type_list = [torch.float, torch.half]
        for input_t in type_list:
            if input_t is torch.half:
                input_self_cpu = torch.normal(mean=20,
                                              std=torch.randn(20, dtype=torch.float))
            else:
                input_self_cpu = torch.normal(mean=20,
                                              std=torch.randn(20, dtype=torch.float))
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_ptr = input_self_mlu.data_ptr()
            input_self_cpu -= 2.3
            input_self_mlu -= 2.3
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_rsub(self):
        a = torch.randn(3, dtype=torch.float)
        b = torch.randn(3, dtype=torch.float)
        out = torch.rsub(a, b)
        out_mlu = torch.rsub(a.to('mlu'), b.to('mlu'))
        self.assertTensorsEqual(out, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_rsub_zero_dim(self):
        a = torch.tensor(3.7, dtype=torch.float)
        b = torch.tensor(2.2, dtype=torch.float)
        out = torch.rsub(a, b)
        out_mlu = torch.rsub(a.to('mlu'), b.to('mlu'))
        self.assertTensorsEqual(out, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_exception(self):
        a = torch.randn(3).bool().to('mlu')
        b = torch.randn(3).bool().to('mlu')
        ref_msg = r"^Subtraction, the \`\-\` operator, with two bool tensors is not supported\. "
        ref_msg = ref_msg + r"Use the \`\^\` or \`logical_xor\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.sub(a, b)

        a = torch.randn(3).bool().to('mlu')
        b = torch.randn(3).to('mlu')
        ref_msg = r"^Subtraction, the \`\-\` operator, with a bool tensor is not supported\. "
        ref_msg = ref_msg + r"If you are trying to invert a mask, use the \`\~\` or "
        ref_msg = ref_msg + r"\`logical\_not\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.sub(a, b)

        a = torch.randn(3).bool().to('mlu')
        b = torch.randn(3).bool().to('mlu')
        ref_msg = r"^Subtraction, the \`\-\` operator, with two bool tensors is not supported\. "
        ref_msg = ref_msg + r"Use the \`\^\` or \`logical_xor\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.rsub(a, b)

        a = torch.randn(3).bool().to('mlu')
        b = torch.randn(3).to('mlu')
        ref_msg = r"^Subtraction, the \`\-\` operator, with a bool tensor is not supported\. "
        ref_msg = ref_msg + r"If you are trying to invert a mask, use the \`\~\` or "
        ref_msg = ref_msg + r"\`logical\_not\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.rsub(a, b)

        a = torch.randn(3).bool().to('mlu')
        b = torch.randn(3).to('mlu')
        ref_msg = r"^Subtraction, the \`\-\` operator, with two bool tensors is not supported\. "
        ref_msg = ref_msg + r"Use the \`\^\` or \`logical\_xor\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.rsub(a, True, 1)

if __name__ == '__main__':
    unittest.main()
