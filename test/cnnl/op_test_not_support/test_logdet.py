from __future__ import print_function
import sys
import os

os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF'  # pylint: disable=C0413
import unittest
import copy

import torch
import torch_mlu.core.mlu_model as ct  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

import logging  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestLogdetOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_logdet_basic(self):
        shape_list = [(1, 4, 4), (3, 3), (4, 5, 3, 3)]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.rand(shape, dtype=type, requires_grad=True)
                x_copy = copy.deepcopy(x)
                x_mlu = x_copy.to('mlu')
                out_cpu = torch.logdet(x)
                out_mlu = torch.logdet(x_mlu)
                out_cpu.backward(out_cpu)
                out_mlu.backward(out_mlu)

                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.view(-1)
                    mlu_res = out_mlu.cpu().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(cpu_res[i],
                                            mlu_res[i],
                                            1e-5,
                                            use_MSE=True)
                out_cpu = x.grad
                out_mlu = x_copy.grad
                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.view(-1)
                    mlu_res = out_mlu.cpu().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(cpu_res[i],
                                            mlu_res[i],
                                            1e-5,
                                            use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_logdet_channelslast(self):
        shape_list = [(4, 3, 3, 3)]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype=type, requires_grad=True)
                x_copy = copy.deepcopy(x)
                x_mlu = x_copy.to('mlu')
                out_cpu = torch.logdet(x.permute(0, 2, 3, 1))
                out_mlu = torch.logdet(x_mlu.permute(0, 2, 3, 1))
                out_cpu.backward(out_cpu)
                out_mlu.backward(out_mlu)
                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.view(-1)
                    mlu_res = out_mlu.cpu().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(cpu_res[i],
                                            mlu_res[i],
                                            1e-5,
                                            use_MSE=True)
                out_cpu = x.grad
                out_mlu = x_copy.grad
                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.view(-1)
                    mlu_res = out_mlu.cpu().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(cpu_res[i],
                                            mlu_res[i],
                                            1e-5,
                                            use_MSE=True)


if __name__ == '__main__':
    unittest.main()
