from __future__ import print_function

import sys
import os
import itertools
import copy
import unittest
import logging

import torch
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

def to_mlu(tensor_cpu):
    return tensor_cpu.to(ct.mlu_device())


class TestMulOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_mul(self):
        data_types = [torch.float, torch.half]
        for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 224)),
                               ((1, 3, 224, 224), (1, 3, 1, 1)),
                               ((1, 1, 24, 1), (1, 1, 24, 1)), ((10), (1)),
                               ((1, 3, 224, 1), (1, 3, 1, 224)),
                               ((1, 3, 224, 224), (1, 1, 1, 1))]:
            for data_type in data_types:
                a = torch.rand(shape1, dtype=torch.float)
                b = torch.rand(shape2, dtype=torch.float)

                out_cpu = a * b
                out_mlu = self.to_mlu_dtype(a, data_type) * self.to_mlu_dtype(b, data_type)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul_tensor_tensor_channel_last(self):
        """
        test_tensor_tensor
        """
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[..., ::2]]
        param_list = [dtype_list, func_list, func_list]
        #for data_type, err in dtype_list:
        for data_type, func_x, func_y in itertools.product(*param_list):
            for shape1, shape2 in [((224, 224), (1, 10, 224, 1)),
                                   ((1, 10, 224, 224), (1, 10, 224, 1))]:
                a = torch.rand(shape1).to(data_type)
                b = torch.rand(shape2).to(data_type)

                out_cpu = func_x(a) * func_y(b)
                out_mlu = func_x(a.to("mlu")) * func_y(b.to("mlu"))

                # float type precision : 0.003
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                        3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul_inplace(self):
        for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 224)),
                               ((1, 2, 3, 4), (1, 3, 4)),
                               ((2, 2, 7, 8), (7, 8))]:
            x1 = to_mlu(torch.ones(shape1, dtype=torch.float))
            x1_half = self.to_mlu_dtype(torch.ones(shape1, dtype=torch.float), torch.half)
            x2 = torch.ones(shape1, dtype=torch.float)
            y = torch.rand(shape2, dtype=torch.float)
            raw_ptr = x1.data_ptr()
            x1.mul_(to_mlu(y))
            x2.mul_(y)
            self.assertEqual(raw_ptr, x1.data_ptr())
            self.assertTensorsEqual(x2, x1.cpu(), 3e-3, use_MSE=True)

            raw_ptr = x1_half.data_ptr()
            x1_half.mul_(self.to_mlu_dtype(y, torch.half))
            self.assertEqual(raw_ptr, x1_half.data_ptr())
            self.assertTensorsEqual(x2, x1_half.cpu().float(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_scalar(self):
        data_types = [torch.float, torch.half]
        for shape in [(224), (2, 4, 5, 3), (24, 24)]:
            for data_type in data_types:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 * b.sum()
                out_mlu = 1.2 * b.sum().to(data_type).to(ct.mlu_device())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul_inplace_intscalar(self):
        type_list = [
            torch.float, torch.int, torch.short, torch.int8, torch.uint8,
            torch.long, torch.half, torch.double
        ]
        for input_t in type_list:
            if input_t is torch.half:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float))
                input_self_mlu = copy.deepcopy(input_self_cpu).to(input_t).to(ct.mlu_device())
            else:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float)).to(input_t)
                input_self_mlu = copy.deepcopy(input_self_cpu).to(ct.mlu_device())
            input_self_cpu *= 1
            input_self_mlu *= 1
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul_inplace_floatscalar(self):
        data_types = [torch.float, torch.half]
        for data_type in data_types:
            input_self_cpu = torch.normal(mean=20,
                                          std=torch.randn(20, dtype=torch.float))
            input_self_mlu = copy.deepcopy(input_self_cpu).to(data_type).to(ct.mlu_device())
            input_self_cpu *= 2.3
            input_self_mlu *= 2.3
            self.assertTensorsEqual(input_self_cpu.float(),
                                    input_self_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul_scalar_dtype(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half, torch.double
        ]
        for scalar in [3, 3.3, True]:
            for type_t in type_list:
                if type_t is torch.half:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float))
                    input_self_mlu = input_self_cpu.to(type_t).to(ct.mlu_device())
                else:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float)).to(type_t)
                    input_self_mlu = input_self_cpu.to(ct.mlu_device())
                out_cpu = input_self_cpu * scalar
                out_mlu = input_self_mlu * scalar
                if type_t is torch.half:
                    self.assertEqual(out_mlu.dtype, torch.half)
                else:
                    self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                if out_cpu.dtype == torch.bool:
                    for val in range(out_cpu[0]):
                        self.assertEqual(out_cpu[val].item(),
                                         out_mlu.cpu()[val].item())
                else:
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3 if type_t is torch.half else 0.0,
                                            use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_scalar_mul_dtype(self):
        type_list = [
            torch.bool, torch.float, torch.int, torch.short, torch.int8,
            torch.uint8, torch.long, torch.half, torch.double
        ]
        for scalar in [3, 3.3, True]:
            for type_t in type_list:
                if type_t is torch.half:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float))
                    input_self_mlu = input_self_cpu.to(type_t).to(ct.mlu_device())
                elif type_t is torch.uint8:
                  # prevent data overflow
                    input_self_cpu = torch.randperm(n = 63).to(type_t)
                    input_self_mlu = input_self_cpu.to(type_t).to(ct.mlu_device())
                else:
                    input_self_cpu = torch.normal(mean=5,
                        std=torch.randn(5, dtype=torch.float)).to(type_t)
                    input_self_mlu = input_self_cpu.to(ct.mlu_device())
                out_cpu = scalar * input_self_cpu
                out_mlu = scalar * input_self_mlu
                if type_t is torch.half:
                    self.assertEqual(out_mlu.dtype, torch.half)
                else:
                    self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                if out_cpu.dtype == torch.bool:
                    for val in range(out_cpu[0]):
                        self.assertEqual(out_cpu[val].item(),
                                         out_mlu.cpu()[val].item())
                else:
                    self.assertTensorsEqual(out_cpu.float(),
                                            out_mlu.cpu().float(),
                                            3e-3 if type_t is torch.half else 0.0,
                                            use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mul_out(self):
        for shape1, shape2 in [((3, 4, 2), (3, 4, 2))]:
            a = torch.randn(shape1)
            b = torch.randn(shape2)
            out_cpu = torch.randn(shape1)
            torch.mul(a, b, out=out_cpu)
            # the element number of out >= the expected of the op
            out_mlu = self.to_device(torch.randn(shape1))
            torch.mul(self.to_device(a), self.to_device(b), out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)
            # the element number of out < the expected of the op
            out_mlu = self.to_device(torch.randn((1,)))
            torch.mul(self.to_device(a), self.to_device(b), out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

if __name__ == "__main__":
    unittest.main()
