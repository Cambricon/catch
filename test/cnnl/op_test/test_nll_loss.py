from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import copy
import unittest
import logging
from itertools import product

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
    def test_nll_loss_forward(self):
        N_lst = [8, 64, 128]
        C_lst = [20, 1000]
        ignore_lst = [0, 1, -100]
        reduct_lst = ["none", "mean", "sum"]
        weight_lst = [True, False]
        dtype_lst = [torch.float, torch.half]
        product_lst = product(reduct_lst,
                               N_lst,
                               C_lst,
                               ignore_lst,
                               weight_lst,
                               dtype_lst)
        for reduct, N, C, ignore, weight_flag, dtype in product_lst:
            if dtype == torch.half:
                x = torch.randn(N, C, requires_grad=True).to(dtype).to(torch.float)
                # generate weight
                weight = torch.randn(C).abs().to(dtype).to(torch.float)
            else:
                x = torch.randn(N, C, requires_grad=True).to(dtype)
                # generate weight
                weight = torch.randn(C).abs().to(dtype)

            # generate target
            if weight_flag:
                weight_ = weight
                weight_mlu = self.to_device(weight)
            else:
                weight_ = None
                weight_mlu = None
            target = torch.randint(0, C, [N], dtype=torch.long)

            layer = torch.nn.NLLLoss(weight_,
                                     reduction=reduct,
                                     ignore_index=ignore)
            out_cpu = layer(x, target)

            layer_mlu = torch.nn.NLLLoss(weight_mlu,
                                         reduction=reduct,
                                         ignore_index=ignore)
            out_mlu = layer_mlu(self.to_device(x), self.to_device(target))
  
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True)

            # non-contiguous
            target_ = torch.randint(0, C, [N,2], dtype=torch.long)
            out_cpu2 = layer(x, target_[:,0]) 
            out_mlu2 = layer_mlu(self.to_device(x), self.to_device(target_)[:,0])
            self.assertTensorsEqual(out_cpu2.float(), out_mlu2.cpu().float(), 0.003, use_MSE=True)


    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss_backward(self):
        N_lst = [8, 64, 128]
        C_lst = [20, 1000]
        ignore_lst = [0, 1, -100]
        reduct_lst = ["none", "sum", "mean"]
        weight_lst = [True, False]
        dtype_lst = [torch.float, torch.half]
        product_lst = product(reduct_lst,
                               N_lst,
                               C_lst,
                               ignore_lst,
                               weight_lst,
                               dtype_lst)
        for reduct, N, C, ignore, weight_flag, dtype in product_lst:
            if dtype == torch.half:
                x = torch.randn(N, C).to(dtype).to(torch.float)
                weight = torch.randn(C).abs().to(dtype).to(torch.float)
            else:
                x = torch.randn(N, C).to(dtype)
                weight = torch.randn(C).abs().to(dtype)

            x.requires_grad = True
            if weight_flag:
                weight_ = weight
                weight_mlu = self.to_mlu_dtype(weight_, dtype)
            else:
                weight_ = None
                weight_mlu = None

            # generate target
            target = torch.randint(0, C, [N], dtype=torch.long)

            layer = torch.nn.NLLLoss(weight_,
                                     reduction=reduct,
                                     ignore_index=ignore)
            out_cpu = layer(x, target)
            if dtype == torch.half:
                grad = torch.ones(out_cpu.shape).to(dtype).to(torch.float)
            else:
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

            if dtype == torch.half:
                self.assertTensorsEqual(
                    a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.01, use_MSE=True)
            else:
                self.assertTensorsEqual(
                    a_grad_cpu.float(), a_grad_mlu.cpu().float(), 0.003, use_MSE=True)

            # non-contiguous
            if dtype == torch.half:
                x = torch.randn(N, C).to(dtype).to(torch.float)
                weight = torch.randn(C,2).abs().to(dtype).to(torch.float)
            else:
                x = torch.randn(N, C).to(dtype)
                weight = torch.randn(C,2).abs().to(dtype)

            x.requires_grad = True
            if weight_flag:
                weight_ = weight[:,0]
                weight_mlu = self.to_mlu_dtype(weight, dtype)[:,0]
            else:
                weight_ = None
                weight_mlu = None

            target_ = torch.randint(0, C, [N,2], dtype=torch.long)

            layer2 = torch.nn.NLLLoss(weight_,
                                      reduction=reduct,
                                      ignore_index=ignore)
            out_cpu2 = layer2(x, target_[:,0]) 
            if dtype == torch.half:
                grad = torch.ones(out_cpu2.shape).to(dtype).to(torch.float)
            else:
                grad = torch.ones(out_cpu2.shape).to(dtype)
            out_cpu2.backward(grad)
            a_grad_cpu2 = copy.deepcopy(x.grad)

            x.grad.zero_()
            layer_mlu2 = torch.nn.NLLLoss(weight_mlu,
                                          reduction=reduct,
                                          ignore_index=ignore)
            out_mlu2 = layer_mlu2(self.to_mlu_dtype(x, dtype), self.to_device(target_)[:,0])
            out_mlu2.backward(self.to_mlu_dtype(grad, dtype))
            a_grad_mlu2 = copy.deepcopy(x.grad)

            self.assertTensorsEqual(
                out_cpu2, out_mlu2.cpu(), 3e-3, use_MSE=True)
            
            if dtype == torch.half:
                self.assertTensorsEqual(
                    a_grad_cpu2.float(), a_grad_mlu2.cpu().float(), 0.01, use_MSE=True)
            else:
                self.assertTensorsEqual(
                    a_grad_cpu2.float(), a_grad_mlu2.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss_2d(self):
        N, C = 2, 3
        D = 4
        loss = nn.NLLLoss()
        input_t = torch.randn((N,C,D,D), dtype=torch.float, requires_grad=True)
        target = torch.empty(N, D, D, dtype=torch.long).random_(0, C)
        input_copy = copy.deepcopy(input_t)
        input_mlu = input_copy.to('mlu')
        target_mlu = target.to('mlu')
        output = loss(input_t, target)
        output_mlu = loss(input_mlu, target_mlu)
        self.assertTensorsEqual(
            output, output_mlu.cpu(), 3e-3, use_MSE=True)

        grad = torch.randn(output.shape, dtype=torch.float, requires_grad=True)
        grad_mlu = copy.deepcopy(grad).to('mlu')
        output.backward(grad)
        output_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            input_t.grad, input_copy.grad.cpu(), 3e-3, use_MSE=True)
        
        # non-contiguous
        input_t_ = copy.deepcopy(input_t)

        input_copy_ = copy.deepcopy(input_t)
        input_mlu_ = input_copy_.to('mlu')
        target_copy = copy.deepcopy(target)

        output_ = loss(input_t_, target.transpose(1,2).contiguous())
        output_mlu_ = loss(input_mlu_, target_copy.to('mlu').transpose(1,2).contiguous())
        self.assertTensorsEqual(
            output_, output_mlu_.cpu(), 3e-3, use_MSE=True)

        grad_ = torch.randn(output_.shape, dtype=torch.float, requires_grad=True)
        grad_mlu_ = copy.deepcopy(grad_).to('mlu')

        output_.backward(grad_)
        output_mlu_.backward(grad_mlu_)
        self.assertTensorsEqual(
            input_t_.grad, input_copy_.grad.cpu(), 3e-3, use_MSE=True)
        

    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss_exception(self):
        loss = nn.NLLLoss()
        input = torch.randn((10,4), dtype=torch.float, requires_grad=True).to('mlu')
        target = torch.empty((10,5), dtype=torch.long).random_(0, 4).to('mlu')
        ref_msg = r"^1D target tensor expected, multi-target not supported$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            loss(input, target)

        loss = nn.NLLLoss()
        input = torch.randn((10,4), dtype=torch.float, requires_grad=True).to('mlu')
        target = torch.empty(9, dtype=torch.long).random_(0, 4).to('mlu')
        ref_msg = r"^Expected input batch_size \(10\) to match target batch_size \(9\)\.$"
        with self.assertRaisesRegex(ValueError, ref_msg):
            loss(input, target)

        loss = nn.NLLLoss(weight=torch.randn(5, dtype=torch.float).to('mlu'))
        input = torch.randn((10,4), dtype=torch.float, requires_grad=True).to('mlu')
        target = torch.empty((10), dtype=torch.long).random_(0, 4).to('mlu')
        ref_msg = r"^weight tensor should be defined either for all 4 classes or no classes"
        ref_msg = ref_msg + r" but got weight tensor of shape: \[5\]$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            loss(input, target)

if __name__ == "__main__":
    unittest.main()
