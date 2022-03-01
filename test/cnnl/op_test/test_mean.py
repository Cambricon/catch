from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import itertools
import unittest
import logging

import torch
from torch_mlu.core.mlu_model import is_using_floating_device

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_mean_dim(self):
        type_list = [True,False]
        shape_list = [(1,32,5,12,8),(2,128,10,6),(2,512,8),(1,100),(24,)]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[..., ::2]]
        param_list = [type_list, shape_list, func_list]
        for test_type, shape, func in itertools.product(*param_list):
            dim_len = len(shape)
            for i in range(1,dim_len+1):
                dim_lists =  list(itertools.permutations(range(dim_len), i)) + \
                    list(itertools.permutations(range(-dim_len, 0), i))
                for test_dim in dim_lists:
                    x = torch.randn(shape, dtype=torch.float)
                    out_cpu = func(x).mean(test_dim,keepdim=test_type)
                    out_mlu =func(self.to_mlu(x)).mean(test_dim,keepdim = test_type)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(),0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mean(self):
        shape_list = [(2,3,4,3,4,2,1),(2,3,4),(1,32,5,12,8),
                      (2,128,10,6),(2,512,8),(1,100),(24,)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.mean(x)
            out_mlu = torch.mean(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mean_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.mean(x)
        out_mlu = torch.mean(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mean_out(self):
        type_list = [True, False]
        shape_list = [(1,32,5,12,8),(2,128,10,6),(2,512,8),(1,100),(24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len+1):
                dim_lists = list(itertools.permutations(range(0, dim_len), i)) + \
                    list(itertools.permutations(range(-dim_len, 0), i))
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        out_cpu = torch.randn(1)
                        out_mlu = self.to_mlu(torch.randn(1))
                        x_mlu = self.to_mlu(x)
                        torch.mean(x, test_dim, keepdim=test_type, out=out_cpu)
                        torch.mean(x_mlu, test_dim, keepdim=test_type, out=out_mlu)
                        self.assertTensorsEqual(
                            out_cpu.float(), out_mlu.cpu().float(),0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mean_empty(self):
        x = torch.randn(1, 0, 1)
        out_mlu = self.to_mlu(x).mean()
        if is_using_floating_device():
            # MLU370 returns nan. nan != nan itself.
            assert out_mlu.cpu().item() != out_mlu.cpu().item()
        else:
            assert out_mlu.cpu().item() == 0

    #@unittest.skip("not test")
    @testinfo()
    def test_mean_exception(self):
        a = torch.randn((3,4)).int().to('mlu')
        b = torch.tensor(1).to('mlu')
        ref_msg = r"^Can only calculate the mean of floating types\. Got Int instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.mean(dim=1)

        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.mean()

        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.mean(a, out=b, dim=1)

if __name__ == "__main__":
    unittest.main()
