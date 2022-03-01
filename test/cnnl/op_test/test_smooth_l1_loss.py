from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_forward(self):
        shape_list = [(1,),
                      (2,2),
                      (3,7,8),
                      (32, 4, 8732)]
        reduct_list = ["none", "sum", "mean"]
        dtype_list = [torch.float] # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            target = torch.randn(item[0]).to(item[2])

            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(self.to_device(x), target.to(ct.mlu_device()))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_forward_not_dense(self):
        shape_list = [(32, 4, 5, 8732), (5, 3, 2, 3, 30)]
        reduct_list = ["none", "sum", "mean"]
        dtype_list = [torch.float] # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])[:, :, :, :15]
            target = torch.randn(item[0]).to(item[2])[:, :, :, :15]

            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(self.to_device(x), target.to(ct.mlu_device()))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_forward_channels_last(self):
        shape_list = [(32, 4, 5, 8732), (5, 3, 2, 3, 10)]
        reduct_list = ["none", "sum", "mean"]
        dtype_list = [torch.float] # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            x = self.convert_to_channel_last(x)
            target = torch.randn(item[0]).to(item[2])
            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(self.to_device(x), target.to(ct.mlu_device()))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float().contiguous(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_backward_channel_last(self):
        shape_list = [(1,),
                      (32, 4, 8732),
                      (12, 3, 416, 416),
                      (5, 3, 2, 3, 10)]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float] # half is support for mlu
        channel_last = [False, True]
        for item in product(shape_list, reduct_list, dtype_list):
            for channel in channel_last:
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
                y = x
                if channel is False:
                    y = self.convert_to_channel_last(x)
                out_mlu = layer_mlu(self.to_device(y), target.to(ct.mlu_device()))
                out_mlu_ptr = out_mlu.data_ptr()
                out_mlu.backward(grad_output_mlu)
                grad_input_mlu = copy.deepcopy(x.grad)

                self.assertEqual(
                    out_mlu_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float().contiguous(), 0.003, use_MSE=True)
                self.assertTensorsEqual(
                    grad_input_cpu, grad_input_mlu, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_backward(self):
        shape_list = [(1,),
                      (2, 2),
                      (3, 7, 8),
                      (32, 4, 8732),
                      (12, 3, 416, 416),
                      (5, 3, 2, 3, 10)]
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
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(grad_output_mlu)
            grad_input_mlu = copy.deepcopy(x.grad)

            self.assertEqual(
                out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                grad_input_cpu, grad_input_mlu, 0.003, use_MSE=True)

            # not contiguous test
            if len(item[0]) in (3, 4):
                x_ori = torch.randn(item[0], requires_grad=True).to(item[2])
                target_ori = torch.randn(item[0]).to(item[2])
                x = x_ori[:, 1:2, 2:6]
                target = target_ori[:, 1:2, 2:6]
                layer = torch.nn.SmoothL1Loss(reduction=item[1])
                out_cpu = layer(x, target)
                grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
                grad_output_mlu = grad_output.to(torch.device('mlu'))
                out_cpu.backward(grad_output)
                grad_input_cpu = copy.deepcopy(x_ori.grad)

                x_ori.grad.zero_()
                layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
                out_mlu = layer_mlu(self.to_device(x), target.to(ct.mlu_device()))
                out_mlu_ptr = out_mlu.data_ptr()
                out_mlu.backward(grad_output_mlu)
                grad_input_mlu = copy.deepcopy(x_ori.grad)

                self.assertEqual(
                    out_mlu_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
                self.assertTensorsEqual(
                    grad_input_cpu, grad_input_mlu, 0.003, use_RAE=True)

if __name__ == "__main__":
    unittest.main()
