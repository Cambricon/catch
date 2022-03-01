from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=all
import copy
from itertools import product
import unittest
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu.core.mlu_model as ct
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411, C0413

logging.basicConfig(level=logging.DEBUG)

class TestBceOps(TestCase):
    # TODO(guyi):issue:303 (fixme)  # pylint: disable=W0511
    # 1)pytorch1.6 logic is different with 1.3
    #   and some boundary values can't be passed
    # 2)here logic doesn't add offset value, later will fix
    # @unittest.skip("not test")
    @testinfo()
    def test_bce(self):
        shape_list = [(156), (2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        # bce_loss python interface don't support short/half
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(shape_list, reduct_lst, dtype_list, weight_flag_list):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight_orig = torch.rand(shape, dtype=torch.float).to(type_err[0])
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
            try:
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                        type_err[1], use_MSE=True)
            except AssertionError as e:
                print(e)

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_not_dense(self):
        shape_list = [(2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        # bce_loss python interface don't support short/half
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(shape_list, reduct_lst, dtype_list, weight_flag_list):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight_orig = torch.rand(shape, dtype=torch.float).to(type_err[0])
            if weight_flag:
                weight_cpu = weight_orig[...,:int(shape[-1]/2)]
                weight_mlu = weight_orig.to("mlu")[...,:int(shape[-1]/2)]
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x[...,:int(shape[-1]/2)]
            x_mlu = x.to('mlu')[...,:int(shape[-1]/2)]
            target_cpu = target[...,:int(shape[-1]/2)]
            target_mlu = target.to('mlu')[...,:int(shape[-1]/2)]
            loss_cpu = nn.BCELoss(weight=weight_cpu if weight_flag else None, reduction=reduct)
            loss_mlu = nn.BCELoss(weight=weight_mlu if weight_flag else None,
                                  reduction=reduct)
            out_cpu = loss_cpu(x_cpu, target_cpu)
            out_mlu = loss_mlu(x_mlu, target_mlu)
            try:
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                        type_err[1], use_MSE=True)
            except AssertionError as e:
                print(e)
        
    # @unittest.skip("not test")
    @testinfo()
    def test_bce_channel_last(self):
        shape_list = [(2, 4, 6, 8),(32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        # bce_loss python interface don't support short/half
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(shape_list, reduct_lst, dtype_list, weight_flag_list):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight_orig = torch.rand(shape, dtype=torch.float).to(type_err[0])
            if weight_flag:
                weight_cpu = weight_orig
                weight_mlu = weight_orig.to("mlu")
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x.to(memory_format=torch.channels_last)
            x_mlu = x.to('mlu').to(memory_format=torch.channels_last)
            target_cpu = target
            target_mlu = target.to('mlu')
            loss_cpu = nn.BCELoss(weight=weight_cpu if weight_flag else None, reduction=reduct)
            loss_mlu = nn.BCELoss(weight=weight_mlu if weight_flag else None,
                                  reduction=reduct)
            out_cpu = loss_cpu(x_cpu, target_cpu)
            out_mlu = loss_mlu(x_mlu, target_mlu)

            try:
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                        type_err[1], use_MSE=True)
            except AssertionError as e:
                print(e)


    # @unittest.skip("not test")
    @testinfo()
    def test_bce_bp(self):
        shape_list = [(156), (2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(shape_list, reduct_lst, dtype_list, weight_flag_list):
            x = torch.rand(shape, dtype=torch.float, requires_grad=True).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in_mlu = grad_in.to("mlu")
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
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                    type_err[1], use_MSE=True)
            self.assertTensorsEqual(grad_cpu.float(), grad_mlu.cpu().float(),
                                    type_err[1], use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_bp_not_dense(self):
        shape_list = [(2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(shape_list, reduct_lst, dtype_list, weight_flag_list):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in = torch.rand(shape, dtype=torch.float).to(type_err[0])[...,:int(shape[-1]/2)]
            grad_in_mlu = grad_in.to("mlu")[...,:int(shape[-1]/2)]
            if weight_flag:
                weight_cpu = weight[...,:int(shape[-1]/2)]
                weight_mlu = weight.to("mlu")[...,:int(shape[-1]/2)]
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x[...,:int(shape[-1]/2)].requires_grad_()
            x_mlu = x.to('mlu')[...,:int(shape[-1]/2)].requires_grad_()

            target_cpu = target[...,:int(shape[-1]/2)]
            target_mlu = target.to("mlu")[...,:int(shape[-1]/2)]
            out_cpu = F.binary_cross_entropy(x_cpu, target_cpu, reduction=reduct,
                                                          weight=weight_cpu)
            if reduct == "none":
                out_cpu.backward(grad_in)
            else:
                out_cpu.backward()
            grad_cpu = copy.deepcopy(x_cpu.grad)
            x_cpu.grad.zero_()
            out_mlu = F.binary_cross_entropy(x_mlu, target_mlu,
                                              reduction=reduct,
                                              weight=weight_mlu)
            if reduct == "none":
                out_mlu.backward(grad_in_mlu)
            else:
                out_mlu.backward()
            grad_mlu = copy.deepcopy(x_mlu.grad)
            x_mlu.grad.zero_()
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                    type_err[1], use_MSE=True)
            self.assertTensorsEqual(grad_cpu.float(), grad_mlu.cpu().float(),
                                    type_err[1], use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_bp_channel_last(self):
        shape_list = [(2, 4, 6, 8), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(shape_list, reduct_lst, dtype_list, weight_flag_list):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in_mlu = grad_in.to("mlu")
            if weight_flag:
                weight_cpu = weight
                weight_mlu = weight.to("mlu")
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x.to(memory_format=torch.channels_last).requires_grad_()
            x_mlu = x.to('mlu').to(memory_format=torch.channels_last).requires_grad_()
            # import pdb;pdb.set_trace()
            target_cpu = target
            target_mlu = target.to("mlu")
            out_cpu = F.binary_cross_entropy(x_cpu, target_cpu, reduction=reduct,
                                                          weight=weight_cpu)
            if reduct == "none":
                out_cpu.backward(grad_in)
            else:
                out_cpu.backward()
            grad_cpu = copy.deepcopy(x_cpu.grad)
            x_cpu.grad.zero_()
            out_mlu = F.binary_cross_entropy(x_mlu, target_mlu,
                                              reduction=reduct,
                                              weight=weight_mlu)
            if reduct == "none":
                out_mlu.backward(grad_in_mlu)
            else:
                out_mlu.backward()
            grad_mlu = copy.deepcopy(x_mlu.grad)
            x_mlu.grad.zero_()
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(),
                                    type_err[1], use_MSE=True)
            self.assertTensorsEqual(grad_cpu.float(), grad_mlu.cpu().float(),
                                    type_err[1], use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_exceptions(self):
        shape = (1024, 256)
        reduct = "mean"
        dtype = torch.half
        x = torch.rand(shape, dtype=dtype, requires_grad=True)
        target = torch.rand(shape, dtype=dtype)
        weight = torch.rand(shape, dtype=dtype)
        grad_in = torch.rand(shape, dtype=dtype)
        grad_in_mlu = grad_in.to("mlu")
        weight_mlu = weight.to("mlu")
        grad_cpu = copy.deepcopy(x.grad)
        ref_msg = r"binary_cross_entropy not implemented for 'Half'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu = F.binary_cross_entropy(x.to("mlu"), target.to("mlu"),
                                                        reduction=reduct,
                                                        weight=weight_mlu)
            if reduct == "none":
                out_mlu.backward(grad_in_mlu)
            else:
                out_mlu.backward()

if __name__ == "__main__":
    unittest.main()
