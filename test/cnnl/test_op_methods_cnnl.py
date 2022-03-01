# pylint: disable=W0511
from __future__ import print_function  # pylint: disable=C0302
import logging
import sys
import os
import copy
import itertools
import unittest
import math
from itertools import product
from functools import reduce
from operator import mul
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413
import torch_mlu.core.mlu_model as ct  # pylint: disable=C0413
logging.basicConfig(level=logging.DEBUG)
os.environ['TEST_CPU_DISPATCH'] = '1'

class LogSoftMaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, result=None):
        if ct.is_mlu_tensor(x):
            result = torch.log_softmax(x,dim)
        else:
            if result is None:
                logging.error("logsoftmaxbackward requires result!!")
        ctx.save_for_backward(x, result)
        ctx.dim = dim
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x,result = ctx.saved_tensors
        dim = ctx.dim
        grad = torch._log_softmax_backward_data(grad_output, result, dim, x)  # pylint: disable=W0212
        return grad

class Net_cpu(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(Net_cpu, self).__init__()
        self.features = nn.Linear(512, 2048)

    def forward(self, x):
        output = self.features(x)
        return output

class TestOp(TestCase):  # pylint: disable=R0904

    #@unittest.skip("not test")
    @testinfo()
    def test_threshold(self):
        shape_list = [(3, 2)]
        dtype_list = [torch.float32]
        list_list = [shape_list, dtype_list]
        for shape, dtype in product(*list_list):
            m = nn.Threshold(1, 2)
            # test forward
            input_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            output_cpu = m(input_cpu)
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0)
            # test out
            output_mlu = self.to_device(torch.randn(shape, dtype=dtype))
            with torch.no_grad():
                torch.threshold(self.to_device(input_mlu), 1, 2, out=output_mlu)
                self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)

        # test inplace
        m = nn.Threshold(1.1, 2.2, inplace=True)
        input_cpu = torch.randn(shape_list[0])
        input_mlu = input_cpu.to('mlu')
        data_ptr = input_mlu.data_ptr()
        m(input_cpu)
        m(input_mlu)
        self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0)
        self.assertEqual(data_ptr, input_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d(self):
        in_shape_list = [(8, 16, 14, 14)]
        out_shape_list = [(4, 4)]
        dtype_list = [torch.float]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveMaxPool2d(out_shape)
            output_cpu = m(input_cpu.float())
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(),
                                    output_mlu.cpu().float(),
                                    3e-3, use_MSE=True)
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)
            # test out
            output_mlu = self.to_device(torch.randn(output_cpu.size(), dtype=dtype))
            # TODO(zhanchendi): this output index have different meaning for CNNL,
            # so we do not check it currently, may be fixed in higher version of CNNL.
            index_mlu = self.to_device(torch.randint(-10, 10, output_cpu.size()))
            with torch.no_grad():
                torch._C._nn.adaptive_max_pool2d(self.to_device(input_mlu), out_shape,
                  out=[output_mlu, index_mlu])
                self.assertTensorsEqual(output_cpu.float(),
                                        output_mlu.cpu().float(),
                                        3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d(self):
        in_shape_list = [(8, 16, 14, 14)]
        out_shape_list = [(4, 4)]
        dtype_list = [torch.float]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool2d(out_shape)
            output_cpu = m(input_cpu)
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(),
                                    output_mlu.cpu().float(),
                                    3e-3, use_MSE=True)
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)
            # test out
            output_mlu = self.to_device(torch.randn(output_cpu.size(), dtype=dtype))
            with torch.no_grad():
                torch._C._nn.adaptive_avg_pool2d(self.to_device(input_mlu), out_shape,
                  out=output_mlu)
                self.assertTensorsEqual(output_cpu.float(),
                                        output_mlu.cpu().float(),
                                        3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_argmax(self):
        x = torch.tensor([[[ 3,  -8,  -9,   0],
                           [-2,   5, -10,   1],
                           [ 3,  -3,   5,  -3]],
                          [[ 5,   7,   3,   6],
                           [ 5,   4,  -7,  -3],
                           [-3,  -2,  -1, -10]]])
        out_mlu = torch.argmax(self.to_device(x))
        self.assertEqual(out_mlu.cpu().item(), 13)

    #@unittest.skip("not test")
    @testinfo()
    def test_add_inplace_floatscalar(self):
        type_list = [torch.float]
        for input_t in type_list:
            input_self_cpu = torch.randn((2, 3, 4)).to(input_t)
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_self_cpu += 2.3
            input_self_mlu += 2.3
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

            input_self_cpu = input_self_cpu + 2.3
            input_self_mlu = input_self_mlu + 2.3
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_scalar(self):
        dtype_list = [(torch.float, 3e-3)]
        for data_type, err in dtype_list:
            for shape in [(2, 4, 5, 3)]:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 - b.sum()
                out_mlu = 1.2 - self.to_mlu_dtype(b.sum(), data_type)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
                c = torch.rand(shape, dtype=torch.float)
                out_cpu = c.sum() - 1.2
                out_mlu = self.to_mlu_dtype(c.sum(), data_type) - 1.2
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
    #@unittest.skip("not test")
    @testinfo()
    def test_bitwise_or(self):
        shape_list = [(2, 3, 4)]
        dtype_list = [torch.int32]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randint(-16777216, 16777216, shape).type(dtype)
                y = torch.randint(-16777216, 16777216, shape).type(dtype)
                out_cpu = torch.bitwise_or(x, y)
                out_mlu = torch.bitwise_or(self.to_device(x), self.to_device(y))
                out_cpu_1 = x | 1
                out_mlu_1 = self.to_device(x) | 1
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0,
                                        use_MSE = True)
                self.assertTensorsEqual(out_cpu_1.float(), out_mlu_1.cpu().float(), 0.0,
                                        use_MSE = True)
    #@unittest.skip("not test")
    @testinfo()
    def test_bitwise_or_inplace(self):
        shape_list = [(2, 3, 4)]
        dtype_list = [torch.int]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randint(2, shape).type(dtype)
                y = torch.randint(2, shape).type(dtype)
                x_copy = copy.deepcopy(x)
                x_copy_mlu = self.to_device(x_copy)
                out_cpu = x.bitwise_or_(y)
                out_mlu = x_copy_mlu.bitwise_or_(self.to_device(y))
                x_copy |= 1
                x_copy_mlu_1 = self.to_device(x_copy)
                x_copy_mlu_1 |= 1
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0,
                                        use_MSE = True)
                self.assertTensorsEqual(x_copy.float(), x_copy_mlu_1.cpu().float(), 0.0,
                                        use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_bitwise_or_out(self):
        shape_list = [(2, 3, 4)]
        dtype_list = [torch.bool]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randint(2, shape).type(dtype)
                y = torch.randint(2, shape).type(dtype)
                res_cpu = torch.randint(2, shape).type(dtype)
                res_mlu = self.to_device(res_cpu)
                torch.bitwise_or(x, y, out=res_cpu)
                torch.bitwise_or(self.to_device(x), self.to_device(y), out=res_mlu)
                self.assertTensorsEqual(res_cpu.float(), res_mlu.cpu().float(), 0.00,
                                        use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_bitwise_and(self):
        shape_list = [(2, 3, 4)]
        dtype_list = [torch.int32]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randint(-16777216, 16777216, shape).type(dtype)
                y = torch.randint(-16777216, 16777216, shape).type(dtype)
                out_cpu = torch.bitwise_and(x, y)
                out_mlu = torch.bitwise_and(self.to_device(x), self.to_device(y))
                out_cpu_1 = x & 1
                out_mlu_1 = self.to_device(x) & 1
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0,
                                        use_MSE = True)
                self.assertTensorsEqual(out_cpu_1.float(), out_mlu_1.cpu().float(), 0.0,
                                        use_MSE = True)
    #@unittest.skip("not test")
    @testinfo()
    def test_bitwise_and_inplace(self):
        shape_list = [(2, 3, 4)]
        dtype_list = [torch.int]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randint(2, shape).type(dtype)
                y = torch.randint(2, shape).type(dtype)
                x_copy = copy.deepcopy(x)
                x_copy_mlu = self.to_device(x_copy)
                out_cpu = x.bitwise_and_(y)
                out_mlu = x_copy_mlu.bitwise_and_(self.to_device(y))
                x_copy &= 1
                x_copy_mlu_1 = self.to_device(x_copy)
                x_copy_mlu_1 &= 1
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0,
                                        use_MSE = True)
                self.assertTensorsEqual(x_copy.float(), x_copy_mlu_1.cpu().float(), 0.0,
                                        use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_bitwise_and_out(self):
        shape_list = [(2, 3, 4)]
        dtype_list = [torch.bool]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randint(2, shape).type(dtype)
                y = torch.randint(2, shape).type(dtype)
                res_cpu = torch.randint(2, shape).type(dtype)
                res_mlu = self.to_device(res_cpu)
                torch.bitwise_and(x, y, out=res_cpu)
                torch.bitwise_and(self.to_device(x), self.to_device(y), out=res_mlu)
                self.assertTensorsEqual(res_cpu.float(), res_mlu.cpu().float(), 0.00,
                                        use_MSE = True)
    #@unittest.skip("not test")
    @testinfo()
    def test_stack(self):
        # The dimensions of the input tensors must be equal
        shape_list = [(3, 5), (5, 6, 7), (1, 2, 3, 4)]
        # cnnl doesn't support int31, cpu doesn't support float16
        type_list = [torch.float, torch.int8, torch.int16,
                     torch.int64, torch.long, torch.bool]
        for shape in shape_list:
            for type in type_list:
                for dim in [-len(shape)-1, len(shape)]:
                    a_1 = torch.ones(shape, dtype=type)
                    a_2 = torch.ones(shape, dtype=type)
                    out_cpu = torch.stack((a_1, a_2), dim=dim)
                    out_mlu = torch.stack((a_1.to('mlu'), a_2.to('mlu')), dim=dim)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_stack_out(self):
        # The dimensions of the input tensors must be equal
        shape_list = [(3, 5), (5, 6, 7), (1, 2, 3, 4)]
        # cnnl doesn't support int31, cpu doesn't support float16
        type_list = [torch.float, torch.int8, torch.int16,
                     torch.int64, torch.long, torch.bool]
        for shape in shape_list:
            for type in type_list:
                for dim in [-len(shape)-1, len(shape)]:
                    a_1 = torch.ones(shape, dtype=type)
                    a_2 = torch.ones(shape, dtype=type)
                    out_cpu = torch.ones((4, ), dtype=type)
                    out_mlu = out_cpu.to('mlu')
                    torch.stack((a_1, a_2), dim=dim, out=out_cpu)
                    torch.stack((a_1.to('mlu'), a_2.to('mlu')), dim=dim, out=out_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_meshgrid(self):
        shape_lst1 = [(1,), (3,), (7,), (111,)]
        shape_lst2 = [(0,), (2,), (8,), (256,)]
        shape_lst3 = [(5,), (15,), (125,)]
        dtype_lst = [torch.float]
        loop_val = [shape_lst1, shape_lst2, shape_lst3, dtype_lst]
        for param in product(*loop_val):
            shape1, shape2, shape3, dtype = param
            if dtype==torch.float:
                input1 = torch.randn(shape1, dtype=dtype)
                input2 = torch.randn(shape2, dtype=dtype)
                input3 = torch.randn(shape3, dtype=dtype)
            input1_mlu = input1.to(torch.device("mlu"))
            input2_mlu = input2.to(torch.device("mlu"))
            input3_mlu = input3.to(torch.device("mlu"))
            output = torch.meshgrid(input1, input2, input3)
            output_mlu = torch.meshgrid(input1_mlu, input2_mlu, input3_mlu)
            for i in range(3):
                self.assertTensorsEqual(output[i], output_mlu[i].cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp(self):
        shape_list = [(13, 78), (16, 0, 8)]
        data_types = [torch.float]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.exp(x)
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp_(self):
        shape_list = [(13, 78), (16, 0, 8)]
        data_types = [torch.float]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.exp(x)
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_exp_out(self):
        shape_list = [(27), (13, 78), (16, 384, 3072), (13, 24, 35, 46), (16, 0, 8)]
        data_types = [torch.float]
        out_shapes = [(100, 10), (1), (20, 20, 60, 100), (77, 0, 88, 99)]
        for out_shape in out_shapes:
            for shape in shape_list:
                for data_type in data_types:
                    x = torch.randn(shape, dtype=torch.float)
                    x_mlu = self.to_mlu_dtype(x, data_type)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                    torch.exp(x, out=out_cpu)
                    torch.exp(x_mlu, out=out_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul_scalar(self):
        data_types = [torch.float]
        for shape in [(224), (2, 4, 5, 3), (24, 24)]:
            for data_type in data_types:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 * b.sum()
                out_mlu = 1.2 * b.sum().to(data_type).to(ct.mlu_device())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul(self):
        data_types = [torch.float]
        for shape in [(224), (2, 4, 5, 3), (24, 24)]:
            for data_type in data_types:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 * b
                out_mlu = 1.2 * b.to(data_type).to(ct.mlu_device())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_sub_inplace_floatscalar(self):
        type_list = [torch.float]
        for input_t in type_list:
            input_self_cpu = torch.normal(mean=5, std=torch.randn(20, dtype=torch.float))
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_self_cpu -= 2.3
            input_self_mlu -= 2.3
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_pow(self):
        shape_list = [(2, 3, 4),]
        exp_list = [2,]
        data_types = [(torch.float, 3e-3),]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = torch.rand(shape_list[i], dtype=torch.float).abs() + 1
                out_cpu = torch.pow(input1, exp_list[i])
                out_mlu = torch.pow(self.to_mlu_dtype(input1, data_type), exp_list[i])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_pow_inplace(self):
        shape_list = [(2,4),]
        exp_list = [2,]
        data_types = [(torch.float, 3e-3),]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = torch.rand(shape_list[i], dtype=torch.float).abs() + 1
                input1_mlu = self.to_mlu_dtype(input1, data_type)
                input1.pow_(exp_list[i])
                input1_mlu.pow_(exp_list[i])
                self.assertTensorsEqual(
                    input1.float(), input1_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_out(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((5), (5))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2,dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2,dtype=torch.bool).to("mlu")
                out_cpu = torch.eq(x, y, out = out_tmpcpu)
                out_mlu = torch.eq(self.to_mlu(x), self.to_mlu(y), out = out_tmpmlu)
                self.assertTensorsEqual(out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0,
                                        use_MSE=True)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_inplace_scalar(self):
        type_list = [torch.float]
        for t in type_list:
            for shape in [(5)]:
                y = torch.randn(shape).to(t)
                y_mlu = y.to("mlu")
                y.eq_(1.1)
                y_mlu.eq_(1.1)
                self.assertTensorsEqual(y, y_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mm(self):
        x1 = torch.rand((64, 32), dtype=torch.float)
        x2 = torch.rand((32, 100), dtype=torch.float)
        x1_mlu = self.to_mlu(x1)
        x2_mlu = self.to_mlu(x2)
        y_cpu = torch.mm(x1, x2)
        y_mlu = torch.mm(x1_mlu, x2_mlu)
        self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mm_out(self):
        x1 = torch.rand((64, 2048), dtype=torch.float)
        x2 = torch.rand((2048, 1000), dtype=torch.float)
        x1_mlu = self.to_mlu(x1)
        x2_mlu = self.to_mlu(x2)
        y_cpu = torch.zeros(1,)
        y_mlu = y_cpu.to('mlu')
        torch.mm(x1, x2, out=y_cpu)
        torch.mm(x1_mlu, x2_mlu, out=y_mlu)
        self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu(), 0.003, use_MSE=True)
    #@unittest.skip("not test")
    @testinfo()
    def test_all(self):
        shape_list = [(10, 11, 20)]
        dim_list = [-2]
        dtype_list = [torch.bool]
        keep_type = [True]
        for i, list_ in enumerate(shape_list):
            for dtype in dtype_list:
                x = torch.rand(list_, dtype=torch.float)
                x_cpu = x.round().to(dtype)
                x_mlu = self.to_device(x_cpu)
                x_cpu.all(dim = dim_list[i], keepdim = keep_type[i % 2])
                x_mlu.all(dim = dim_list[i], keepdim = keep_type[i % 2])
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.00, use_MSE=True)
                self.assertTrue(x_mlu.dtype == dtype, "all out dtype is not right")
    #@unittest.skip("not test")
    @testinfo()
    def test_all_out(self):
        shape_list = [(10, 11, 20)]
        dim_list = [-2]
        dtype_list = [torch.bool]
        keep_type = [True]
        for i, list_ in enumerate(shape_list):
            for dtype in dtype_list:
                x = torch.rand(list_, dtype=torch.float)
                x_cpu = x.round().to(dtype)
                out_cpu = torch.rand(list_, dtype=torch.float).to(dtype)
                x_mlu = self.to_device(x_cpu)
                out_mlu = self.to_device(out_cpu)
                torch.all(x_cpu, dim = dim_list[i], keepdim = keep_type[i % 2], out=out_cpu)
                torch.all(x_mlu, dim = dim_list[i], keepdim = keep_type[i % 2], out=out_mlu)
                self.assertTrue(out_mlu.dtype == dtype, "all out dtype is not right")
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.00, use_MSE=True)
    #@unittest.skip("not test")
    @testinfo()
    def test_any(self):
        shape_list = [(1, 2, 3, 4)]
        for list_ in shape_list:
            x = torch.rand(list_, dtype=torch.float)
            x_1 = x.bool()
            out_cpu_1 = x_1.any()
            out_mlu_1 = x_1.to("mlu").any()
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.003, use_MSE=True)
            out_cpu_3 = x_1.any(0, keepdim=True)
            out_mlu_3 = x_1.to("mlu").any(0, keepdim=True)
            self.assertTensorsEqual(out_cpu_3, out_mlu_3.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_prod_dim(self):
        type_list = [True, False]
        shape_list = [(2, 512, 8)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = x.prod(item[1], keepdim=item[0])
                out_mlu = self.to_device(x).prod(item[1], keepdim=item[0])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_put(self):
        shapes = [(4, 5)]
        for shape in shapes:
            x = torch.randn(shape, dtype=torch.float)
            scalar_tensor = torch.tensor(0.2)
            x_copy = copy.deepcopy(x)
            mask = torch.randn(shape, dtype=torch.float).bool()
            x[mask] = scalar_tensor
            x_mlu = x_copy.to('mlu')
            mask_mlu = mask.to('mlu')
            x_mlu[mask_mlu] = scalar_tensor.to("mlu")
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.00, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_prod(self):
        shape_list = [(2, 32, 8)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.prod(x)
            out_mlu = torch.prod(self.to_device(x))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_prod_out(self):
        type_list = [False]
        shape_list = [(2, 32, 8)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.randn(1)
                out_mlu = self.to_device(out_cpu)
                x_mlu = self.to_device(x)
                torch.prod(x, item[1], keepdim=item[0], out=out_cpu)
                torch.prod(x_mlu, item[1], keepdim=item[0], out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_gather(self):
        shapes = [(2, 20, 56)]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                index = torch.randint(0, shape[dim], shape)
                out = torch.gather(x, dim, index)
                x_mlu = self.to_mlu(x)
                index_mlu = self.to_device(index)
                out_mlu = torch.gather(x_mlu, dim, index_mlu)
                self.assertTensorsEqual(out, out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_gather_out(self):
        shapes = [(2, 20, 56)]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                index = torch.randint(0, shape[dim], shape)
                out_cpu = torch.randn(shape)
                torch.gather(x, dim, index, out=out_cpu)
                x_mlu = x.to('mlu')
                index_mlu = index.to('mlu')
                out_mlu = torch.randn(shape).to('mlu')
                torch.gather(x_mlu, dim, index_mlu, out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite(self, dtype=torch.float):
        vals = (-float('inf'), float('inf'), float('nan'), -1, 0, 1)
        x = torch.tensor(vals, dtype=dtype)
        res_cpu = torch.isfinite(x)
        res_mlu = torch.isfinite(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())

    #@unittest.skip("not test")
    @testinfo()
    def test_avgpooling_backward(self):
        shape_list = [(8, 16, 7, 7)]
        kernel_v = [2]
        stride_v = [3, None]
        padding_v = [0]
        ceil_mode_v = [False, True]
        include_pad_v = [False]
        loop_var = [shape_list, kernel_v, stride_v, padding_v, ceil_mode_v, include_pad_v]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=torch.float, requires_grad=True)
            avg_pool = nn.AvgPool2d(kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                    count_include_pad=include_pad)
            output_cpu = avg_pool(input_t)
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad)
            grad_cpu = copy.deepcopy(input_t.grad)
            input_t.grad.zero_()
            output_mlu = avg_pool(self.to_device(input_t))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)
            output_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(input_t.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_maxpooling3d(self):
        shape_list = [(12, 128, 8, 112, 112)]
        kernel_v = [(2, 3, 3)]
        stride_v = [(2, 2, 2)]
        padding_v = [(0, 1, 1)]
        ceil_mode_v = [False]
        return_indices_v = [False]

        loop_var = [
            shape_list, kernel_v, stride_v, padding_v, ceil_mode_v, return_indices_v
        ]
        for in_shape, kernel, stride, padding, ceil_mode, return_indices in product(
                *loop_var):
            input_t = torch.randn(in_shape, dtype=torch.float)
            # test nn module
            max_pool = nn.MaxPool3d(kernel,
                                    stride=stride,
                                    padding=padding,
                                    dilation=1,
                                    ceil_mode=ceil_mode,
                                    return_indices=return_indices)
            output_cpu = max_pool(input_t)
            output_mlu = max_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_max_dim(self):
        shape_list = [(2,3,4)]
        dim_list = [1, -1]
        type_list = [True, False]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.max(x, dim_list[i], keepdim=type_list[i])
            out_mlu = torch.max(self.to_device(x), dim_list[i],keepdim=type_list[i])
            self.assertTensorsEqual(out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_max_other(self):
        type_list = [torch.float, torch.int]
        for t in type_list:
            for shape1,shape2 in [((1, 1, 12), (64, 12, 1))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.max(x, y)
                out_mlu = torch.max(self.to_device(x), self.to_device(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_max(self):
        shape_list = [(64, 3, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.max(x)
            out_mlu = torch.max(self.to_device(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_max_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.max(x)
        out_mlu = torch.max(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_maxpooling_backward(self):
        in_shape = (1, 1, 8, 8)
        kernel_v = [3]
        stride_v = [2]
        padding_v = [1]
        ceil_mode_v = [False]
        return_indices_v = [False]
        loop_var = [kernel_v, stride_v, padding_v, ceil_mode_v, return_indices_v]
        for kernel, stride, padding, ceil_mode, return_indices in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=torch.float, requires_grad=True)
            output_cpu = F.max_pool2d(input_t, kernel, stride=stride, padding=padding, dilation=1,
                                      return_indices=return_indices, ceil_mode=ceil_mode)
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad)
            grad_cpu = copy.deepcopy(input_t.grad)
            input_t.grad.zero_()
            output_mlu = F.max_pool2d(self.to_device(input_t),
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu(),
                                    3e-3,
                                    use_MSE=True)
            output_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(input_t.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_maxpooling3d_backward(self):
        shape_list = [(12, 128, 8, 112, 112)]
        kernel_v = [(2, 3, 3)]
        stride_v = [(2, 2, 2)]
        padding_v = [(0, 1, 1)]
        ceil_mode_v = [False]
        return_indices_v = [False]

        loop_var = [
            shape_list, kernel_v, stride_v, padding_v, ceil_mode_v, return_indices_v
        ]
        for in_shape, kernel, stride, padding, ceil_mode, return_indices in product(
                *loop_var):
            input_t = torch.randn(in_shape, dtype=torch.float, requires_grad=True)
            output_cpu = F.max_pool3d(input_t,
                                    kernel,
                                    stride=stride,
                                    padding=padding,
                                    dilation=1,
                                    ceil_mode=ceil_mode,
                                    return_indices=return_indices)
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad)
            grad_cpu = copy.deepcopy(input_t.grad)
            input_t.grad.zero_()
            output_mlu = F.max_pool3d(self.to_device(input_t),
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)
            output_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(input_t.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_local_scalar_dense(self):
        shapes = [(2, 20, 56)]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                index = torch.randint(0, shape[dim], shape)
                out = torch.gather(x, dim, index)
                x_mlu = self.to_mlu(x)
                index_mlu = self.to_device(index)
                out_mlu = torch.gather(x_mlu, dim, index_mlu)
                self.assertTensorsEqual(out, out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_bce_with_logits(self):
        shape_list = [(32, 3, 13, 13)]
        reduct_lst = ["none"]
        for shape in shape_list:
            for reduct in reduct_lst:
                x = torch.rand(shape, dtype=torch.float)
                target = torch.rand(shape, dtype=torch.float)
                weight = torch.rand(shape, dtype=torch.float)
                pos_weight = torch.rand(shape, dtype=torch.float)
                out_cpu = F.binary_cross_entropy_with_logits(x, target, reduction=reduct,
                    weight=weight, pos_weight=pos_weight)
                out_mlu = F.binary_cross_entropy_with_logits(x.to("mlu"),
                    target.to("mlu"), reduction=reduct, weight=weight.to("mlu"),
                    pos_weight=pos_weight.to("mlu"))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_bce_with_logits_bp(self):
        shape_list = [(32, 3, 13, 13)]
        reduct_lst = ["none"]
        for shape in shape_list:
            for reduct in reduct_lst:
                x = torch.rand(shape, dtype=torch.float, requires_grad=True)
                target = torch.rand(shape, dtype=torch.float)
                weight = torch.rand(shape, dtype=torch.float)
                pos_weight = torch.rand(shape, dtype=torch.float)
                grad_in = torch.rand(shape, dtype=torch.float)
                grad_in_mlu = grad_in.to("mlu")
                weight_ = weight
                weight_mlu = weight.to("mlu")
                pos_weight_ = pos_weight
                pos_weight_mlu = pos_weight.to("mlu")
                out_cpu = F.binary_cross_entropy_with_logits(x, target, reduction=reduct,
                                                             weight=weight_,
                                                             pos_weight=pos_weight_)
                out_cpu.backward(grad_in)
                grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()
                out_mlu = F.binary_cross_entropy_with_logits(x.to("mlu"), target.to("mlu"),
                                                            reduction=reduct,
                                                            weight=weight_mlu,
                                                            pos_weight=pos_weight_mlu)
                out_mlu.backward(grad_in_mlu)
                grad_mlu = copy.deepcopy(x.grad)
                x.grad.zero_()
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
    #@unittest.skip("not test")
    @testinfo()
    def test_bce(self):
        shape_list = [(2, 4, 6, 8)]
        reduct_lst = ["none"]
        dtype = [torch.float]
        for shape in shape_list:
            for reduct in reduct_lst:
                for type in dtype:
                    x = torch.rand(shape, dtype=torch.float).to(type)
                    target = torch.rand(shape, dtype=torch.float).to(type)
                    weight_orig = torch.rand(shape, dtype=torch.float).to(type)
                    for weight_flag in [True]:
                        if weight_flag:
                            weight_ = weight_orig
                            weight_mlu = weight_orig.to("mlu")
                        else:
                            weight_ = None
                            weight_mlu = None
                        loss = nn.BCELoss(weight=weight_ if weight_flag else None, reduction=reduct)
                        loss_mlu = nn.BCELoss(weight=weight_mlu if weight_flag else None,
                                              reduction=reduct)
                        out_cpu = loss(x, target)
                        out_mlu = loss_mlu(x.to("mlu"), target.to("mlu"))
                        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
    #@unittest.skip("not test")
    @testinfo()
    def test_bce_bp(self):
        shape_list = [(2, 4, 6, 8)]
        reduct_lst = ["none"]
        for shape in shape_list:
            for reduct in reduct_lst:
                x = torch.rand(shape, dtype=torch.float, requires_grad=True)
                target = torch.rand(shape, dtype=torch.float)
                weight = torch.rand(shape, dtype=torch.float)
                grad_in = torch.rand(shape, dtype=torch.float)
                grad_in_mlu = grad_in.to("mlu")
                for weight_flag in [True]:
                    if weight_flag:
                        weight_ = weight
                        weight_mlu = weight.to("mlu")
                    else:
                        weight_ = None
                        weight_mlu = None
                    out_cpu = F.binary_cross_entropy(x, target, reduction=reduct,
                                                                 weight=weight_)
                    if reduct == "none":
                        out_cpu.backward(grad_in)
                    else:
                        out_cpu.backward()
                    grad_cpu = copy.deepcopy(x.grad)
                    x.grad.zero_()
                    out_mlu = F.binary_cross_entropy(x.to("mlu"), target.to("mlu"),
                                                                reduction=reduct,
                                                                weight=weight_mlu)
                    if reduct == "none":
                        out_mlu.backward(grad_in_mlu)
                    else:
                        out_mlu.backward()
                    grad_mlu = copy.deepcopy(x.grad)
                    x.grad.zero_()
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
    #@unittest.skip("not test")
    @testinfo()
    def test_alias(self):
        dim = 64
        x = torch.randn((dim, dim), dtype=torch.float)
        out_cpu = x[:dim, :dim]
        x_mlu = x.to(ct.mlu_device())
        out_mlu = x_mlu[:dim, :dim]
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_masked_select(self):
        shapes = [(100, 512)]
        dtype = [torch.float]
        for type_ in dtype:
            for shape in shapes:
                x = torch.rand(shape, dtype=type_)
                mask = torch.ones(shape, dtype=bool)
                out_cpu = torch.masked_select(x, mask)
                x_mlu = self.to_device(x)
                mask_mlu = self.to_device(mask)
                out_mlu = torch.masked_select(x_mlu, mask_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.00)

    #@unittest.skip("not test")
    @testinfo()
    def test_masked_select_out(self):
        shapes = [(100, 512)]
        out_shapes = [(512)]
        dtype = [torch.float]
        for type_ in dtype:
            for shape, out_shape in zip(shapes, out_shapes):
                x = torch.rand(shape, dtype=type_)
                mask = torch.ones(shape, dtype=bool)
                out_cpu = torch.rand(out_shape, dtype=type_)
                torch.masked_select(x, mask, out=out_cpu)
                x_mlu = self.to_device(x)
                mask_mlu = self.to_device(mask)
                out_mlu = out_cpu.to(torch.device('mlu'))
                torch.masked_select(x_mlu, mask_mlu, out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.00)

    #@unittest.skip("not test")
    @testinfo()
    def test_masked_fill_tensor(self):
        shapes = [(100, 512)]
        for shape in shapes:
            x = torch.rand(shape, dtype=torch.float)
            mask = torch.ones(shape, dtype=torch.bool)
            value = torch.tensor(2.33, dtype=torch.float)
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            value_mlu = self.to_device(value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float(), 0.00, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_masked_fill_scalar(self):
        shapes = [(100, 512)]
        for shape in shapes:
            x = torch.rand(shape, dtype=torch.float)
            mask = torch.ones(shape, dtype=torch.bool)
            value = 3.14159
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.00, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_empty_strided(self):
        shape_stride_list = [((2, 3), (1, 2)), ((6, 7, 8), (1, 4, 2))]
        for shape, stride in shape_stride_list:
            # Currently MLU Tensor doesn't support stride mode, MLU tensor will
            # be always created in contiguous mode whether stride is set or not.
            x = torch.empty_strided(shape, stride, device=ct.mlu_device())
            x_cpu = torch.empty_strided(shape, stride)
            self.assertEqual(x_cpu.size(), x.size())
            self.assertEqual(x_cpu.stride(), x.stride())

    #@unittest.skip("not test")
    @testinfo()
    def test_nll_loss_forward(self):
        N_lst = [4]
        C_lst = [3]
        ignore_lst = [0]
        reduct_lst = ["mean"]
        for reduct in reduct_lst:
            for N in N_lst:
                for C in C_lst:
                    for ignore in ignore_lst:
                        x = torch.randn(N, C, dtype=torch.float)
                        weight = torch.randn(C, dtype=torch.float).abs()
                        weight_mlu = weight.to("mlu")
                        target = abs(np.random.randn(N))
                        target /= np.max(target)
                        target *= (C - 1)
                        target = np.round(target)
                        target = torch.from_numpy(target).long()
                        layer = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
                        out_cpu = layer(x, target)
                        layer_mlu = torch.nn.NLLLoss(weight_mlu, reduction=reduct,
                                                    ignore_index=ignore)
                        out_mlu = layer_mlu(self.to_mlu(x), target.to(ct.mlu_device()))
                        print(out_cpu, out_mlu.cpu())
                        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_nll_loss_backward(self):
        N_lst = [8]
        C_lst = [20]
        ignore_lst = [0]
        reduct_lst = ["mean"]
        dtype_lst = [torch.float]
        product_lst = product(reduct_lst, N_lst, C_lst, ignore_lst, dtype_lst)
        for reduct, N, C, ignore, dtype in product_lst:
            x = torch.randn(N, C).to(dtype)
            weight = torch.randn(C).abs().to(dtype)
            x.requires_grad = True
            weight_ = weight
            weight_mlu = self.to_mlu_dtype(weight_, dtype)

            # generate target
            target = torch.randint(0, C, [N], dtype=torch.long)
            layer = torch.nn.NLLLoss(weight_,
                                     reduction=reduct,
                                     ignore_index=ignore)
            out_cpu = layer(x, target)
            grad = torch.ones(out_cpu.shape).to(dtype)
            out_cpu.backward(grad)
            a_grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()
            layer_mlu = torch.nn.NLLLoss(weight_mlu,
                                         reduction=reduct,
                                         ignore_index=ignore)
            out_mlu = layer_mlu(self.to_mlu_dtype(x, dtype), target.to(ct.mlu_device()))
            out_mlu.backward(self.to_mlu_dtype(grad, dtype))
            a_grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(
                a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_nll_loss_2d(self):
        N, C = 10, 4
        D = 5
        input_t = torch.randn((N,C,D,D), dtype=torch.float, requires_grad=True)
        target = torch.empty(N, D, D, dtype=torch.long).random_(0, C)
        input_mlu = input_t.to('mlu')
        target_mlu = target.to('mlu')
        loss = nn.NLLLoss()
        output = loss(input_t, target)
        output_mlu = loss(input_mlu, target_mlu)
        self.assertTensorsEqual(output, output_mlu.cpu(), 3e-3, use_MSE=True)
        grad = torch.randn(output.shape, dtype=torch.float, requires_grad=True)
        output.backward(grad)
        grad_cpu = input_t.grad
        input_t.grad.zero_()
        output_mlu.backward(grad.to("mlu"))
        self.assertTensorsEqual(grad_cpu, input_t.grad.cpu(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_forward(self):
        shape_list = [(3,7,8)]
        reduct_list = ["none"]
        dtype_list = [torch.float]
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            target = torch.randn(item[0]).to(item[2])
            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(self.to_device(x), target.to(ct.mlu_device()))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_zero_(self):
        shape_list = [(5, 6, 7)]
        for shape in shape_list:
            a = torch.rand(shape, dtype=torch.float)
            b = copy.deepcopy(a)
            result_cpu = a.zero_()
            b_mlu = self.to_mlu(b)
            result_mlu = b_mlu.zero_()
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_backward(self):
        shape_list = [(3,7,8)]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float] # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            target = torch.randn(item[0]).to(item[2])
            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
            grad_output_mlu = grad_output.to(torch.device('mlu'))
            out_cpu.backward(grad_output)
            grad_input_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(self.to_device(x), target.to(ct.mlu_device()))
            out_mlu.backward(grad_output_mlu)
            grad_input_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(grad_input_cpu, grad_input_mlu, 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_backward(self):
        shapes = [(16, 5, 7)]
        log_softmax_cpu = LogSoftMaxFunc()
        log_softmax_mlu = LogSoftMaxFunc()
        for shape in shapes:
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_mlu = log_softmax_mlu.apply(self.to_mlu(x), dim)
                out_cpu = log_softmax_cpu.apply(x, dim, out_mlu.cpu())
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                grad_cpu = out_cpu.grad_fn.apply(grad)
                grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
                self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_bmm(self):
        for shape_a, shape_b in [((3, 4, 5), (3, 5, 6))]:
            a = torch.randint(-3, 3, shape_a, dtype=torch.int8)
            b = torch.randint(-3, 3, shape_b, dtype=torch.int8)
            a = a.to(torch.float)
            b = b.to(torch.float)
            a_mlu = self.to_mlu(copy.deepcopy(a))
            b_mlu = self.to_mlu(copy.deepcopy(b))
            out_cpu = torch.bmm(a, b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_matmul(self):
        mat1_shape = (8, 9)
        mat2_shape = (9, 11)
        # other type don't be supported in CNNL
        # and matmul use adaptive_quantize, only can use float as input
        mat1 = torch.randn(mat1_shape, dtype=torch.float, requires_grad=True)
        mat2 = torch.randn(mat2_shape, dtype=torch.float, requires_grad=True)

        out_cpu = torch.matmul(mat1, mat2)
        out_mlu = torch.matmul(mat1.to(ct.mlu_device()), mat2.to(ct.mlu_device()))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

        grad = torch.randn(out_cpu.shape)
        out_cpu.backward(grad)
        grad1 = copy.deepcopy(mat1.grad)
        grad2 = copy.deepcopy(mat2.grad)

        mat1.grad.zero_()
        mat2.grad.zero_()

        grad_mlu = grad.to(ct.mlu_device())
        mat1_mlu = mat1.to(ct.mlu_device())
        mat2_mlu = mat2.to(ct.mlu_device())

        out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
        out_mlu.backward(grad_mlu)

        self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
        self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_lt_inplace(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y.lt_(x)
                y_mlu.lt_(x_mlu)
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)
                y.lt_(1.1)
                y_mlu.lt_(1.1)
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_lt_out(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2,dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2,dtype=torch.bool).to("mlu")
                out_cpu = torch.lt(x, y, out = out_tmpcpu)
                out_mlu = torch.lt(self.to_mlu(x), self.to_mlu(y), out = out_tmpmlu)
                self.assertTensorsEqual(out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0,
                use_MSE=True)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = torch.lt(x, 1.1, out = out_tmpcpu)
                out_mlu = torch.lt(self.to_mlu(x), 1.1, out = out_tmpmlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_leaky_relu_inplace(self):
        for in_shape in [(5, 6)]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            input_cpu = copy.deepcopy(input_)
            negative_slopes = [0.01]
            input_mlu = input_.to("mlu")
            for nega_val in negative_slopes:
                F.leaky_relu(input_cpu, inplace=True, negative_slope=nega_val)
                F.leaky_relu(input_mlu, inplace=True, negative_slope=nega_val)
                self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_softmax_backward(self):
        shapes = [(2, 3, 4)]
        for shape in shapes:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = F.softmax(x, 0)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu = F.softmax(self.to_mlu(x), 0)
            out_mlu.backward(self.to_mlu(grad))
            self.assertTensorsEqual(grad_cpu, x.grad.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_leaky_relu_backward(self):
        for shape in [(9, 17)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = F.leaky_relu(x)
            out_mlu = F.leaky_relu(x.to("mlu"))
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu.backward(grad.to("mlu"))
            grad_mlu = x.grad
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0003)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.0003)

    #@unittest.skip("not test")
    @testinfo()
    def test_cat_out(self):
        x = torch.randn((24, ), dtype=torch.float)
        x_mlu = self.to_mlu(x.clone())
        out_cpu = torch.randn((4, ), dtype=torch.float)
        out_mlu = self.to_mlu(torch.randn((4, ), dtype=torch.float))
        torch.cat([x[:2], x[4:6]], out=out_cpu)
        torch.cat([x_mlu[:2], x_mlu[4:6]], out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_log_out(self):
        shape_list = [(2, 3, 4)]
        out_shape_list = [(24)]
        for shape, out_shape in zip(shape_list, out_shape_list):
            x = torch.rand(shape) + 0.0001
            out_cpu = torch.randn(out_shape)
            out_mlu = self.to_device(torch.randn(out_shape))
            torch.log(x, out=out_cpu)
            torch.log(self.to_device(x), out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_le_inplace(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((5), (5))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y.le_(x)
                y_mlu.le_(x_mlu)
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)
                y.le_(1.1)
                y_mlu.le_(1.1)
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_le_out(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((5), (5))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2,dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2,dtype=torch.bool).to("mlu")
                out_cpu = torch.le(x, y, out = out_tmpcpu)
                out_mlu = torch.le(self.to_mlu(x), self.to_mlu(y), out=out_tmpmlu)
                self.assertTensorsEqual(out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0,
                use_MSE=True)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = torch.le(x, 1.1, out = out_tmpcpu)
                out_mlu = torch.le(self.to_mlu(x), 1.1, out=out_tmpmlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_le(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((5), (5))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.le(x, y)
                out_mlu = torch.le(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = x==y
                out_mlu = self.to_mlu(x) == self.to_mlu(y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_ge_inplace(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y.ge_(x)
                y_mlu.ge_(x_mlu)
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)
                y.ge_(1.1)
                y_mlu.ge_(1.1)
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_ge(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((5), (5))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.ge(x, y)
                out_mlu = torch.ge(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = x==y
                out_mlu = self.to_mlu(x) == self.to_mlu(y)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

                out_cpu = torch.ge(x, 1.1)
                out_mlu = torch.ge(self.to_mlu(x), 1.1)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_ge_out(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2,dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2,dtype=torch.bool).to("mlu")
                out_cpu = torch.ge(x, y, out = out_tmpcpu)
                out_mlu = torch.ge(self.to_mlu(x), self.to_mlu(y), out=out_tmpmlu)
                self.assertTensorsEqual(out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0,
                use_MSE=True)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = torch.ge(x, 1.1, out = out_tmpcpu)
                out_mlu = torch.ge(self.to_mlu(x), 1.1, out=out_tmpmlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_neg(self):
        shape_list = [(2, 3, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.neg(x)
            out_mlu = torch.neg(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_inplace(self):
        type_list = [torch.float]
        shape = (2, 3, 4)
        for Type in type_list:
            out_cpu = torch.randn(shape, dtype=Type)
            out_mlu = out_cpu.to("mlu")
            out_cpu.sigmoid_()
            out_mlu.sigmoid_()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sigmoid_backward(self):
        for in_shape in [(1, 3, 16, 16)]:
            x = torch.randn(in_shape, dtype=torch.float, requires_grad=True)
            x_mlu = x.to(ct.mlu_device())
            out_cpu = x.sigmoid()
            out_mlu = x_mlu.sigmoid()
            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.to(ct.mlu_device())
            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu().float(), 0.003)

    #@unittest.skip("not test")
    @testinfo()
    def test_sum_out(self):
        type_list = [True]
        shape_list = [(2,16,8)]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len+1):
                dim_lists = list(itertools.permutations(range(dim_len), i)) + \
                    list(itertools.permutations(range(-dim_len, 0), i))
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        out_cpu = torch.randn(shape)
                        out_mlu = self.to_device(torch.randn(shape))
                        x_mlu = self.to_device(x)
                        torch.sum(x, test_dim, keepdim=test_type, out=out_cpu)
                        torch.sum(x_mlu, test_dim, keepdim=test_type, out=out_mlu)
                        self.assertTensorsEqual(
                            out_cpu.float(), out_mlu.cpu().float(),0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_eq_inplace(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((5), (5))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y.eq_(x)
                y_mlu.eq_(x_mlu)
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_squeeze(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for in_shape in [(2, 1, 2, 1, 2), (2, 3, 4)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape, dtype=torch.float)
                output_cpu = torch.squeeze(input1)
                output_mlu = torch.squeeze(self.to_mlu_dtype(input1, data_type))
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    #@unittest.skip("not test")
    @testinfo()
    def test_squeeze_inplace(self):
        for in_shape in [(2, 2, 1, 2)]:
            for dim in [1]:
                input_t = torch.randn(in_shape, dtype=torch.float)
                input_mlu = copy.deepcopy(input_t).to('mlu')
                input_t.squeeze_(dim)
                input_mlu.squeeze_(dim)
                self.assertTensorsEqual(input_t, input_mlu.cpu(), 0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_max_out(self):
        for shape1,shape2 in [((1, 1, 10), (64, 10, 1))]:
            x = torch.randn(shape1, dtype=torch.float)
            y = torch.randn(shape2, dtype=torch.float)
            out_cpu = torch.randn(1, dtype=torch.float)
            x_mlu = x.to('mlu')
            y_mlu = y.to('mlu')
            out_mlu = out_cpu.to('mlu')
            torch.max(x, y, out=out_cpu)
            torch.max(x_mlu, y_mlu, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_upsample_nearest2d_out(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype=type)
                out_cpu = torch.randn(1, dtype=type)
                out_mlu = torch.randn(1, dtype=type).to('mlu')
                output_size =  [int(math.floor(x.size(i + 2))) for i in range(2)]
                torch._C._nn.upsample_nearest2d(x, output_size, out=out_cpu) # pylint: disable=I1101, W0212
                torch._C._nn.upsample_nearest2d(x.to('mlu'), output_size, out=out_mlu)  # pylint: disable=I1101, W0212
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_upsample_bilinear2d(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
        align_corners = [True, False]
        type_list = [torch.float32]
        for type in type_list:
            for corner in align_corners:
                for shape in shape_list:
                    m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=corner)
                    x = torch.randn(shape, dtype=type, requires_grad=True)
                    out_cpu = m(x)
                    out_mlu = m(self.to_mlu(x))

                    grad = torch.randn(out_cpu.shape, dtype=torch.float)
                    out_cpu.backward(grad)
                    grad_cpu = copy.deepcopy(x.grad)
                    x.grad.zero_()
                    out_mlu.backward(self.to_device(grad))
                    grad_mlu = copy.deepcopy(x.grad)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_clone(self):
        a = torch.randn(1, 3, 512, 224, dtype=torch.float)
        b = torch.zeros(1, 3, 512, 224, dtype=torch.float)
        a_mlu = a.to(ct.mlu_device())
        b_mlu = b.to(ct.mlu_device())
        out_cpu = a.clone()
        out_mlu = a_mlu.clone()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        c = a + b
        c_mlu = a_mlu + b_mlu
        out_cpu = c.clone()
        out_mlu = c_mlu.clone()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_upsample_nearest2d(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
        m = nn.UpsamplingNearest2d(scale_factor=2)
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype=type, requires_grad=True)
                out_cpu = m(x)
                out_mlu = m(self.to_mlu(x))

                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()
                out_mlu.backward(self.to_device(grad))
                grad_mlu = copy.deepcopy(x.grad)

                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_min_out(self):
        for shape1,shape2 in [((1, 1, 10), (64, 10, 1))]:
            x = torch.randn(shape1, dtype=torch.float)
            y = torch.randn(shape2, dtype=torch.float)
            out_cpu = torch.randn(1, dtype=torch.float)
            x_mlu = x.to('mlu')
            y_mlu = y.to('mlu')
            out_mlu = out_cpu.to('mlu')
            torch.min(x, y, out=out_cpu)
            torch.min(x_mlu, y_mlu, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_l1l2norm(self):
        shape_list = [(2,3,4)]
        scalar_ops_list = [(2)]
        dim_list = [(-1),(0)]
        keep_list = [True, False]
        loop_var = [shape_list, scalar_ops_list, dim_list, keep_list]
        for shape, scalar_op, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = x.norm(scalar_op, dim, keepdim = keep)
            out_mlu =self.to_mlu(x).norm(scalar_op, dim, keepdim = keep)
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),0.003, use_MSE = True)

        for shape, scalar_op, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = x.norm(scalar_op)
            out_mlu =self.to_device(x).norm(scalar_op)
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_l1l2norm_dtype_mode(self):
        shape_list = [(2,3,4,3,2)]
        scalar_ops_list = [(2)]
        # only support calculate the mean of floating types
        type_list = [torch.float]
        loop_var = [shape_list, scalar_ops_list, type_list]
        for shape, scalar_op, type in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = torch.norm(x, p=scalar_op, dtype = type)
            out_mlu = torch.norm(self.to_mlu(x), p=scalar_op, dtype = type)
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_l1l2norm_scalar(self):
        x = torch.tensor(5.2, dtype = torch.float)
        scalar_ops_list = [(1),(2)]
        data_types = [torch.float]
        for scalar_op in scalar_ops_list:
            for data_type in data_types:
                out_cpu = x.norm(scalar_op)
                out_mlu = self.to_mlu_dtype(x, data_type).norm(scalar_op)
                self.assertTensorsEqual(out_cpu.float(),
                                        out_mlu.cpu().float(),
                                        0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_min_dim(self):
        shape_list = [(2,3,4)]
        dim_list = [1]
        type_list = [True]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.min(x, dim_list[i], keepdim=type_list[i])
            out_mlu = torch.min(self.to_mlu(x), dim_list[i],keepdim=type_list[i])
            self.assertTensorsEqual(out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True)
            # min sorting algorithm for mlu is different from cpu,
            # when value is the same the min index may be different,
            # in this case, index test is not included for min in unit test.

    #@unittest.skip("not test")
    @testinfo()
    def test_min_other(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1,shape2 in [((1, 1, 1024), (64, 1024, 1))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.min(x, y)
                out_mlu = torch.min(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_min(self):
        shape_list = [(64, 3, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.min(x)
            out_mlu = torch.min(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_min_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.min(x)
        out_mlu = torch.min(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_sort(self):
        shape_list = [(76, 102)]
        type_list = [torch.short]
        for i in shape_list:
            local_value = [j for j in range(len(i))]  # pylint: disable=R1721
            for typeId in type_list:
                x = torch.randn(i, dtype=torch.float).to(typeId)
                for dim in local_value:
                    out_cpu = torch.sort(x, dim, descending=False)
                    out_mlu = torch.sort(x.to("mlu"), dim, descending=False)
                    self.assertTensorsEqual(
                        out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True)
                    self.assertTrue(out_cpu[1].dtype, out_mlu[1].cpu().dtype)
                    self.assertTensorsEqual(
                        out_cpu[1].float(), out_mlu[1].cpu().float(), 1.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_sort_out(self):
        shape_list = [(76, 124)]
        for i in shape_list:
            x = torch.randn(i, dtype=torch.float)
            local_value = [-j for j in range(len(i))]
            for dim in local_value:
                for descending_true in [False, True]:
                    values = torch.randn(i, dtype=torch.float)
                    indices = torch.randint(-10, 10, i)
                    torch.sort(x, dim, descending=descending_true, out=(values, indices))
                    values_mlu = values.to("mlu")
                    indices_mlu = indices.to("mlu")
                    torch.sort(x.to("mlu"), dim, descending=descending_true, out=(values_mlu,
                                   indices_mlu))
                    self.assertTensorsEqual(
                            values.float(), values_mlu.cpu().float(), 0.0, use_MSE=True)
                    self.assertTrue(indices.dtype, indices_mlu.cpu().dtype)
                    self.assertTensorsEqual(
                            indices.float(), indices_mlu.cpu().float(), 1.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_round(self):
        shape_list = [(2,3,4),]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.round(x)
            out_mlu = torch.round(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_inplace(self):
        shape_list = [(2,3,4)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to('mlu')
            out_cpu = torch.round_(x_cpu)
            out_mlu = torch.round_(x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_out(self):
        shape_list = [(2,3,4)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to('mlu')
            out_tmpcpu = torch.zeros(shape_list[i])
            out_tmpmlu = torch.zeros(shape_list[i]).to('mlu')
            out_cpu = torch.round(x_cpu, out=out_tmpcpu)
            out_mlu = torch.round(x_mlu, out=out_tmpmlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_mean_out(self):
        type_list = [True]
        shape_list = [(1,100)]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len+1):
                dim_lists = list(itertools.permutations(range(0, dim_len), i)) + \
                    list(itertools.permutations(range(-dim_len, 0), i))
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        out_cpu = torch.randn(shape)
                        out_mlu = self.to_mlu(torch.randn(shape))
                        x_mlu = self.to_mlu(x)
                        torch.mean(x, test_dim, keepdim=test_type, out=out_cpu)
                        torch.mean(x_mlu, test_dim, keepdim=test_type, out=out_mlu)
                        self.assertTensorsEqual(
                            out_cpu.float(), out_mlu.cpu().float(),0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_maxpooling_index(self):
        in_shape = (1, 2, 12, 12)
        input_t = torch.randn(in_shape, dtype=torch.float)
        kernel_v = [2]
        stride_v = [3]
        padding_v = [0, 1]
        ceil_mode_v = [False]
        return_indices_v = [True]
        loop_var = [kernel_v, stride_v, padding_v, ceil_mode_v]
        for kernel, stride, padding, ceil_mode in product(*loop_var):
            output_cpu = F.max_pool2d(input_t,
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices_v,
                                      ceil_mode=ceil_mode)
            output_mlu = F.max_pool2d(self.to_mlu(input_t),
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices_v,
                                      ceil_mode=ceil_mode)
            self.assertTensorsEqual(output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_abs_out(self):
        shape_list = [(2, 3, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to(ct.mlu_device())
            out_cpu = torch.abs(x, out=y)
            out_mlu = torch.abs(self.to_mlu(x), out=y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_linspace(self):
        start_list = [1, 3, 3.5]
        end_list = [2, 5, 2.5]
        steps_list = [3, 1, 1]
        for i in range(len(start_list)):  # pylint: disable=C0200
            x = torch.linspace(start_list[i], end_list[i], steps=steps_list[i], device="cpu")
            x_mlu = torch.linspace(start_list[i], end_list[i], steps=steps_list[i], device="mlu")
            self.assertTensorsEqual(x, x_mlu.cpu(), 1e-7, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_linspace_out(self):
        start_list = [1, 3, 3.5]
        end_list = [2, 5, 2.5]
        steps_list = [3, 1, 1]
        for i in range(len(start_list)):  # pylint: disable=C0200
            in1 = torch.randn(1, dtype=torch.float)
            x = torch.linspace(start_list[i], end_list[i], steps=steps_list[i],
                                    device="cpu", out=in1)
            x_mlu = torch.linspace(start_list[i], end_list[i], steps=steps_list[i],
                                    out=in1.to("mlu"))
            self.assertTensorsEqual(x, x_mlu.cpu(), 1e-7, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_gt_inplace(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y.gt_(x)
                y_mlu.gt_(x_mlu)
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)
                y.gt_(1.1)
                y_mlu.gt_(1.1)
                self.assertTensorsEqual(y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_gt_out(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2,dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2,dtype=torch.bool).to("mlu")
                out_cpu = torch.gt(x, y, out = out_tmpcpu)
                out_mlu = torch.gt(self.to_mlu(x), self.to_mlu(y), out = out_tmpmlu)
                self.assertTensorsEqual(out_tmpcpu, out_tmpmlu.cpu().float(), 0.0, use_MSE=True)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = torch.gt(x, 1.1, out = out_tmpcpu)
                out_mlu = torch.gt(self.to_mlu(x), 1.1, out = out_tmpmlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_ne_inplace(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y.ne_(x)
                y_mlu.ne_(x_mlu)
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)
                y.ne_(1.1)
                y_mlu.ne_(1.1)
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_ne_out(self):
        type_list = [torch.float]
        for t in type_list:
            for shape1, shape2 in [((1), (256, 7))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape2,dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2,dtype=torch.bool).to("mlu")
                out_cpu = torch.ne(x, y, out = out_tmpcpu)
                out_mlu = torch.ne(self.to_mlu(x), self.to_mlu(y), out=out_tmpmlu)
                self.assertTensorsEqual(out_tmpcpu, out_tmpmlu.cpu().float(), 0.0, use_MSE=True)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.0, use_MSE=True)
                out_cpu = torch.ne(x, 1.1, out = out_tmpcpu)
                out_mlu = torch.ne(self.to_mlu(x), 1.1, out=out_tmpmlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_sub_inplace(self):
        input_self_cpu = torch.rand(1, 3, 2, 2, dtype=torch.float)
        input_self_mlu = self.to_mlu(input_self_cpu)
        input_other = torch.rand(1, 3, 2, 2, dtype=torch.float)
        out_cpu = input_self_cpu.sub_(input_other)
        out_mlu = input_self_mlu.sub_(self.to_mlu(input_other))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

        out_cpu = input_self_cpu.sub_(1.1, 2)
        out_mlu = input_self_mlu.sub_(1.1, 2)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_sub(self):
        for shape1, shape2 in [((10, 30, 50, 20), (10, 30, 50, 20))]:
            input_self = torch.rand(shape1, dtype=torch.float)
            input_other = torch.rand(shape2, dtype=torch.float)
            out_cpu = input_self - input_other
            out_mlu = self.to_mlu(input_self) - self.to_mlu(input_other)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 5e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_sub_scale(self):
        input_self = torch.rand(1, 3, 15, 15, dtype=torch.float)
        out_cpu = input_self.sub(0.1)
        out_mlu = self.to_mlu(input_self).sub(0.1)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_softplus_backward(self):
        for shape in [(1, 3, 16, 16)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = F.softplus(x,beta=1, threshold=20)
            out_mlu = F.softplus(self.to_device(x),beta=1, threshold=20)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x_cpu = copy.deepcopy(x)
            x.grad.zero_()
            out_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(x, x_cpu, 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_softplus(self):
        for in_shape in [(8, 24, 24)]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            output_cpu = F.softplus(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.softplus(self.to_device(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, input_, 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_tanh_backward(self):
        in_shape = [(35, 46)]
        type_list = [torch.float]
        for shape in in_shape:
            for typeId in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(typeId)
                x_mlu = x.to(ct.mlu_device())
                out_cpu = x_0.tanh()
                out_mlu = x_mlu.tanh()
                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.to(ct.mlu_device())
                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)
                self.assertTensorsEqual(out_grad_cpu,
                    out_grad_mlu.cpu().float() if typeId == torch.half else out_grad_mlu.cpu(),
                    0.003,
                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_gelu_backward(self):
        in_shape = [(35, 46)]
        type_list = [torch.float]
        for shape in in_shape:
            for typeId in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(typeId)
                x_mlu = x.to(ct.mlu_device())
                out_cpu = F.gelu(x_0)
                out_mlu = F.gelu(x_mlu)
                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.to(ct.mlu_device())
                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)
                self.assertTensorsEqual(out_grad_cpu,
                    out_grad_mlu.cpu().float() if typeId == torch.half else out_grad_mlu.cpu(),
                    0.003,
                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_hardtanh_backward(self):
        shape_list = [(45, 50)]
        minmax_list = [(-0.2, 0.4)]
        for shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in [torch.float]:
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
                    self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_floor_out(self):
        shape_list = [(2, 3, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(1, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to(torch.device('mlu'))

            out_cpu = torch.floor(x, out=y)
            out_mlu = torch.floor(self.to_mlu(x), out=y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_fill(self):
        shape_list =[(2, 4, 5)]
        for shape in shape_list:
            x = torch.randn(shape, dtype = torch.float)
            x_mlu = copy.deepcopy(x).to('mlu')
            index = torch.tensor([0, 2])
            index_mlu = index.to('mlu')
            x.index_fill_(1, index, -1)
            x_mlu.index_fill_(1, index_mlu, -1)
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_fill_(self):
        shape_list = [(2, 3, 4)]
        value_list = [5]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            input1 = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.fill_(input1, value_list[i])
            out_mlu_1 = torch.fill_(self.to_mlu(input1), value_list[i])
            out_mlu_2 = torch.fill_(self.to_mlu(
                input1), self.to_mlu(torch.tensor(value_list[i])))
            out_mlu_3 = self.to_mlu(input1).fill_(value_list[i])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu_1.cpu().float(), 0.000, use_MSE=True)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu_2.cpu().float(), 0.000, use_MSE=True)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu_3.cpu().float(), 0.000, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_addmm(self):
        tensor_dtype_list = [torch.float]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((2, 6), (2, 3), (3, 6)),
                      ((22, 58), (22, 45), (45, 58)), ((0, 50), (0, 20), (20, 50)),
                      ((20), (10, 33), (33, 20)), ((13), (4, 0), (0, 13))]
        for tensor_dtype in tensor_dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, tensor_dtype)
                m1_mlu = self.to_mlu_dtype(m1, tensor_dtype)
                m2_mlu = self.to_mlu_dtype(m2, tensor_dtype)
                res_cpu = torch.addmm(M, m1, m2)
                res_mlu = torch.addmm(M_mlu, m1_mlu, m2_mlu)
                self.assertTensorsEqual(res_cpu, res_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_(self):
        tensor_dtype_list = [torch.float]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((2, 6), (2, 3), (3, 6)),
                      ((22, 58), (22, 45), (45, 58)), ((0, 50), (0, 20), (20, 50)),
                      ((4, 13), (4, 0), (0, 13))]
        for tensor_dtype in tensor_dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, tensor_dtype)
                m1_mlu = self.to_mlu_dtype(m1, tensor_dtype)
                m2_mlu = self.to_mlu_dtype(m2, tensor_dtype)
                M.addmm_(m1, m2)
                M_mlu.addmm_(m1_mlu, m2_mlu)
                self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_out(self):
        tensor_dtype_list = [torch.float]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((2, 6), (2, 3), (3, 6)),
                      ((22, 58), (22, 45), (45, 58)), ((0, 50), (0, 20), (20, 50)),
                      ((20), (10, 33), (33, 20)), ((13), (4, 0), (0, 13))]
        out_shapes = [(100, 10), (1), (20, 20, 60, 100)]
        for tensor_dtype in tensor_dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for out_shape in out_shapes:
                    M = torch.randn(M_shape, dtype=torch.float)
                    m1 = torch.randn(m1_shape, dtype=torch.float)
                    m2 = torch.randn(m2_shape, dtype=torch.float)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), tensor_dtype)
                    M_mlu = self.to_mlu_dtype(M, tensor_dtype)
                    m1_mlu = self.to_mlu_dtype(m1, tensor_dtype)
                    m2_mlu = self.to_mlu_dtype(m2, tensor_dtype)
                    torch.addmm(M, m1, m2, out=out_cpu)
                    torch.addmm(M_mlu, m1_mlu, m2_mlu, out=out_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)


    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero(self):
        shape_list = [(2, 3, 4, 5)]
        dtype_list = [torch.float32]
        for dtype in dtype_list:
            for shape in shape_list:
                a = torch.randint(3, shape).type(dtype)
                result_cpu = a.nonzero()
                result_mlu = self.to_device(a).nonzero()
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            # test scalar input
            a = torch.tensor(0).type(dtype)
            result_cpu = a.nonzero()
            result_mlu = self.to_device(a).nonzero()
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_out(self):
        a = torch.randint(3, (2, 2, 3)).type(torch.bool)
        # the element number of out >= the expected of the op
        out_cpu = torch.randint(3, (a.numel() * a.dim(),))
        out_mlu = self.to_device(torch.randint(3, (a.numel() * a.dim(),)))
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

        # test scalar input
        a = torch.tensor(1).type(torch.bool)
        out_cpu = torch.randint(3, (1,))
        out_mlu = self.to_device(torch.randint(3, (1,)))
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_clamp(self):
        for shape1, shape2 in [((5), (5))]:
            for min_ in (0.1, None):
                for max_ in (10, None):
                    if max_ is None and min_ is None:
                        continue
                    x = torch.randn(shape1, dtype = torch.float)
                    out_cpu = torch.clamp(x, min_, max_)
                    out_mlu = torch.clamp(x.to("mlu"), min_, max_)
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            0.0,
                                            use_MSE=True)

                    # test clamp.out
                    y = torch.randn(shape2, dtype=torch.float)
                    out_cpu = torch.clamp(x, min_, max_, out = y)
                    out_mlu = torch.clamp(x.to("mlu"), min_, max_, out = y.to("mlu"))
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            0.0,
                                            use_MSE=True)

                    # test inplace operation
                    x_cpu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x)
                    x_cpu.clamp_(min_, max_)
                    x_mlu.clamp_(min_, max_)
                    self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_transpose(self):
        shape_lst = [(126, 24, 1024), (4, 12, 45, 100)]
        dim0_lst = [0, 1, 2]
        dim1_lst = [0, 1, 2]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for dim0 in dim0_lst:
                for dim1 in dim1_lst:
                    for data_type, err in data_types:
                        x = torch.randn(shape, dtype=torch.float)
                        x_mlu = self.to_mlu_dtype(x, data_type)
                        output_cpu = x.transpose(dim0, dim1)
                        output_mlu = x_mlu.transpose(dim0, dim1)
                        self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    #@unittest.skip("not test")
    @testinfo()
    def test_t(self):
        shape_lst = [(3, 44), (6, 123), (45, 100), (23), ()]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for data_type, err in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                output_cpu = x.t()
                output_mlu = x_mlu.t()
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    #@unittest.skip("not test")
    @testinfo()
    def test_reciprocal(self):
        dtype_list = [(torch.float, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (4)]:
                x_cpu = torch.rand(shape1, dtype=data_type) + 0.00005
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
        dtype_list = [(torch.float, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (4)]:
                x_cpu = torch.rand(shape1, dtype=data_type) + 0.00005
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
        dtype_list = [(torch.float, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (4)]:
                x_cpu = torch.rand(shape1, dtype=data_type) + 0.00005
                x_mlu = self.to_mlu_dtype(x_cpu, data_type) # pylint: disable=W0612
                x_mlu_ptr = x_mlu.data_ptr()

                out_cpu = torch.zeros(shape1, dtype=data_type)
                out_mlu = torch.zeros(shape1, dtype=data_type).to("mlu")
                torch.reciprocal(x_cpu, out=out_cpu)
                torch.reciprocal(x_mlu, out=out_mlu)

                self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                        err, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
