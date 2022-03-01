from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import unittest
import logging
from itertools import product
import itertools

import torch
import torch.nn as nn
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_mse_loss_special(self):
        loss = torch.nn.functional.mse_loss
        reduction_type = ['none', 'sum', 'mean']
        for reduction in reduction_type:
            # case1
            input_cpu = torch.tensor([])
            other_cpu = torch.tensor([])
            input_device = input_cpu.to("mlu")
            other_device = other_cpu.to("mlu")
            ret_cpu = loss(input_cpu, other_cpu, reduction = reduction)
            ret_device = loss(input_device, other_device, reduction = reduction)
            if 'mean' == reduction:
                assert ret_cpu.isnan().item() == True
                assert ret_device.cpu().isnan().item() == True
            else:
                self.assertEqual(ret_device.cpu(), ret_cpu.cpu())
            # case2
            input_cpu = torch.randn(2,0,3)
            other_cpu = torch.randn(2,0,3)
            input_device = input_cpu.to("mlu")
            other_device = other_cpu.to("mlu")
            ret_cpu = loss(input_cpu, other_cpu, reduction = reduction)
            ret_device = loss(input_device, other_device, reduction = reduction)
            if 'mean' == reduction:
                assert ret_cpu.isnan().item() == True
                assert ret_device.cpu().isnan().item() == True
            else:
                self.assertEqual(ret_device.cpu(), ret_cpu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_mse_loss_exception(self):
        not_support_dtype_list = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
        # loss = torch._C._nn.mse_loss
        loss = torch.nn.MSELoss(reduction="mean")
        for dtype in not_support_dtype_list:
            input  = torch.tensor([1], dtype = dtype).to("mlu")
            other = torch.tensor([1], dtype = dtype).to("mlu")
            ref_msg = "^cnnl_mse_loss is not implemented for"
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                loss(input, other)

    # @unittest.skip("not test")
    @testinfo()
    def test_mseloss(self):
        shape_list = [(3, 5), (3, 5, 7), (4, 3, 16, 16), (4, 3, 8, 8, 7)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_lst = [torch.float, torch.half]
        product_lst = product(shape_list,
                              reduct_lst,
                              dtype_lst)
        for shape, reduct, dtype in product_lst:
            if dtype == torch.half:
                x1 = torch.randn(shape).to(dtype).to(torch.float)
                x2 = torch.randn(shape).to(dtype).to(torch.float)
            else:
                x1 = torch.randn(shape).to(dtype)
                x2 = torch.randn(shape).to(dtype)
            x1.requires_grad=True
            layer = torch.nn.MSELoss(reduction=reduct)
            out_cpu = layer(x1, x2)
            grad = torch.ones(out_cpu.shape).to(dtype)
            out_cpu.backward(grad)
            a_grad_cpu = copy.deepcopy(x1.grad)

            x1.grad.zero_()

            out_mlu = layer(self.to_device(x1), self.to_device(x2))
            out_mlu.backward(self.to_mlu_dtype(grad, dtype))
            a_grad_mlu = copy.deepcopy(x1.grad)

            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mseloss_channel_last(self):
        shape_list = [(4, 3, 16, 16)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_lst = [torch.float, torch.half]
        memory_format_list = [torch.channels_last]
        product_lst = product(shape_list,
                              reduct_lst,
                              dtype_lst,
                              memory_format_list)
        for shape, reduct, dtype, memory_format in product_lst:
            if dtype == torch.half:
                x1 = torch.randn(shape).to(dtype).to(torch.float, memory_format = memory_format)
                x2 = torch.randn(shape).to(dtype).to(torch.float, memory_format = memory_format)
            else:
                x1 = torch.randn(shape, dtype = dtype).to(memory_format = memory_format)
                x2 = torch.randn(shape).to(dtype, memory_format = memory_format)
            x1.requires_grad=True
            layer = torch.nn.MSELoss(reduction=reduct)
            out_cpu = layer(x1, x2)
            grad = torch.randn_like(out_cpu)
            out_cpu.backward(grad)
            a_grad_cpu = copy.deepcopy(x1.grad)

            x1.grad.zero_()

            out_mlu = layer(self.to_device(x1), self.to_device(x2))
            out_mlu.backward(self.to_mlu_dtype(grad, dtype))
            a_grad_mlu = copy.deepcopy(x1.grad)

            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mse_memory_format_combination(self):
        dtype_list = [torch.float, torch.half]
        reduct_lst = ["none", "mean", "sum"]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[:, :, :, :1]]
        param_list = [dtype_list, reduct_lst, func_list, func_list]
        shape = (4, 3, 16, 16)
        for data_type, reduct, func_x, func_y in itertools.product(*param_list):
            x1 = torch.randn(shape, dtype=torch.float)
            x2 = torch.randn(shape, dtype=torch.float)
            x1.requires_grad=True
            layer = torch.nn.MSELoss(reduction=reduct)
            out_cpu = layer(func_x(x1), x2)
            grad = torch.randn_like(out_cpu)
            out_cpu.backward(grad)
            a_grad_cpu = copy.deepcopy(x1.grad)

            x1.grad.zero_()
        
            out_mlu = layer(func_x(self.to_mlu_dtype(x1, data_type)),
                            self.to_mlu_dtype(x2, data_type))
            out_mlu.backward(self.to_mlu_dtype(grad, data_type))
            a_grad_mlu = copy.deepcopy(x1.grad)

            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.003, use_MSE=True)


    # @unittest.skip("not test")
    @testinfo()
    def test_mseloss_broadcast(self):
        shape_list = [((3, 273), (1, 273)),
                      ((2, 2, 4, 2), (1, 2)),
                      ((1, 3, 224, 224), (1, 1, 1))
                      ]
        reduct_lst = ["none", "mean", "sum"]
        dtype_lst = [torch.half]
        product_lst = product(shape_list,
                              reduct_lst,
                              dtype_lst)
        for shape, reduct, dtype in product_lst:
            if dtype == torch.half:
                x1 = torch.randn(shape[0]).to(dtype).to(torch.float)
                x2 = torch.randn(shape[1]).to(dtype).to(torch.float)
            else:
                x1 = torch.randn(shape[0]).to(dtype)
                x2 = torch.randn(shape[1]).to(dtype)
            x1.requires_grad=True
            layer = torch.nn.MSELoss(reduction=reduct)
            out_cpu = layer(x1, x2)
            grad = torch.ones(out_cpu.shape).to(dtype)
            out_cpu.backward(grad)
            a_grad_cpu = copy.deepcopy(x1.grad)

            x1.grad.zero_()

            out_mlu = layer(self.to_device(x1.to(dtype)), self.to_device(x2.to(dtype)))
            out_mlu.backward(self.to_mlu_dtype(grad, dtype))
            a_grad_mlu = copy.deepcopy(x1.grad)

            if "sum" == reduct and out_mlu.dtype == torch.half:
                # cuda will fail in sum reduction mode and torch.float16 dtype, so skip compare
                return

            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nn_scalars_reductions(self):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_reduction_scalars(input, reduction, output):
            if reduction != 'none' or input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [()]:
            for reduction in ['none', 'mean', 'sum']:
                module = torch.nn.MSELoss
                input = torch.randn(input_shape, device="mlu", requires_grad=True)
                target = torch.empty(input_shape).random_(2).to("mlu")

                m = module(reduction=reduction)
                output = m(input, target)
                verify_reduction_scalars(input, reduction, output)

    # @unittest.skip("not test")
    @testinfo()
    def test_mseloss_not_dense(self):
        shape_list = [(4, 3, 16, 16)]
        reduct_lst = ["none", "mean", "sum"]
        memory_format_list = [torch.channels_last]
        product_lst = product(shape_list,
                              reduct_lst,
                              memory_format_list)
        for shape, reduct, memory_format in product_lst:
            x1 = torch.randn(shape).to(memory_format = memory_format)
            x2 = torch.randn(shape).to(memory_format = memory_format)
            x1.requires_grad=True
            x1_mlu = copy.deepcopy(x1)
            layer = torch.nn.MSELoss(reduction=reduct)
            out_cpu = layer(x1[:2, ...], x2[:2, ...])
            out_mlu = layer(self.to_device(x1_mlu)[:2, ...], self.to_device(x2)[:2, ...])
            if reduct == "none":
                o_shape = out_cpu.size()
                grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1).to(memory_format = memory_format)
                out_cpu.backward(grad[..., :-1])
                out_mlu.backward(self.to_mlu(grad)[..., :-1])
            else:
                grad = torch.ones(out_cpu.shape)
                out_cpu.backward(grad)
                out_mlu.backward(self.to_mlu(grad))
            a_grad_cpu = x1.grad
            a_grad_mlu = x1_mlu.grad

            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
