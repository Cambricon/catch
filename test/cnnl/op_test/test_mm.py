from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging
from itertools import product
import torch
from torch_mlu.core.mlu_model import is_using_floating_device as float_dev

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

def shape_gen(ns, ms, ps, t1=False, t2=False):
    shape_a = []
    shape_b = []
    for n, m, p in product(ns, ms, ps):
        if t1:
            shape_a.append((m, n))
        else:
            shape_a.append((n, m))
        if t2:
            shape_b.append((p, m))
        else:
            shape_b.append((m, p))
    return zip(shape_a, shape_b)

ns = [0, 4, 20, 2048]
ms = [0, 5, 45, 512]
ps = [0, 8, 64, 1999]
dtype_err = [(torch.half, 0.03), (torch.float, 0.003),
             (torch.double, 0.003), (torch.int8, 0.003),
             (torch.int16, 0.003), (torch.int, 0.003)] if float_dev() \
            else [(torch.half, 0.03), (torch.float, 0.003)]  # Quantize only support float/half

class TestMMOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_mm(self):
        mem_func = [lambda x:x, self.to_non_dense]
        for (dt, err), (shape_a, shape_b) in product(dtype_err, shape_gen(ns, ms, ps)):
            for mem_func1, mem_func2 in product(mem_func, mem_func):
                x1 = torch.rand(shape_a).to(dt).float()
                x2 = torch.rand(shape_b).to(dt).float()
                x1_mlu = mem_func1(x1.to('mlu').to(dt))
                x2_mlu = mem_func2(x2.to('mlu').to(dt))
                y_cpu = torch.mm(x1, x2)
                y_mlu = torch.mm(x1_mlu, x2_mlu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mm_out(self):
        mem_func = [lambda x:x, self.to_non_dense]
        for (dt, err), (shape_a, shape_b) in product(dtype_err, shape_gen(ns, ms, ps)):
            for mem_func1, mem_func2, mem_func3 in product(mem_func, mem_func, mem_func):
                x1 = torch.rand(shape_a).to(dt).float()
                x2 = torch.rand(shape_b).to(dt).float()
                x1_mlu = mem_func1(x1.to('mlu').to(dt))
                x2_mlu = mem_func2(x2.to('mlu').to(dt))
                y_cpu = torch.zeros(15, 7)  # An out tensor with non-equal shape nor dtype
                y_mlu = mem_func3(y_cpu.to('mlu'))
                torch.mm(x1, x2, out=y_cpu)
                torch.mm(x1_mlu, x2_mlu, out=y_mlu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_mm_exception(self):
        m1 = torch.rand((2, 3, 3), dtype=torch.float).to('mlu')
        m2 = torch.rand((3, 4), dtype=torch.float).to('mlu')
        ref_msg = r"^Expected 2-dimensional tensor, but got " + str(m1.dim()) \
                  + r"-dimensional tensor for argument \#1 \'mat1\' \(while" \
                  + r" checking arguments for mm\)$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.mm(m1, m2)

        m1 = torch.rand((2, 3), dtype=torch.float).to('mlu')
        m2 = torch.rand((1, 4), dtype=torch.float).to('mlu')
        ref_msg = r"size mismatch, m1: \[2, 3\], m2: \[1, 4\]"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.mm(m1, m2)

        m1 = torch.randn(1, 1).to(torch.long).to('mlu')
        m2 = torch.randn(1, 1).to('mlu')
        with self.assertRaises(RuntimeError) as M:
            torch.mm(m1, m2)
        msg = "MM mlu op not implemented for dtype of input1: 'long int'"\
              if float_dev() else "quantize_param only support input float/half"
        self.assertEqual(M.exception.args[0], msg)

        m1 = torch.randn(1, 1).to('mlu')
        m2 = torch.randn(1, 1).to(torch.uint8).to('mlu')
        with self.assertRaises(RuntimeError) as M:
            torch.mm(m1, m2)
        msg = "MM mlu op not implemented for dtype of input2: 'unsigned char'"\
              if float_dev() else "quantize_param only support input float/half"
        self.assertEqual(M.exception.args[0], msg)

    # @unittest.skip("not test")
    @testinfo()
    def test_trans_mm(self):
        mem_func = [lambda x:x, self.to_non_dense]
        for t1, t2 in [(True, False), (False, True), (True, True)]:
            for (dt, err), (shape_a, shape_b) \
                in product(dtype_err, shape_gen(ns, ms, ps, t1, t2)):
                for mem_func1, mem_func2 in product(mem_func, mem_func):
                    f = lambda x, t: (x.t() if t else x)
                    x1 = torch.rand(shape_a).to(dt).float()
                    x2 = torch.rand(shape_b).to(dt).float()
                    x1_mlu = mem_func1(x1.to('mlu').to(dt))
                    x2_mlu = mem_func2(x2.to('mlu').to(dt))
                    y_cpu = torch.mm(f(x1, t1), f(x2, t2))
                    y_mlu = torch.mm(f(x1_mlu, t1), f(x2_mlu, t2))
                    self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

if __name__ == '__main__':
    unittest.main()
