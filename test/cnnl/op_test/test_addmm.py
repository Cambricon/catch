from __future__ import print_function
import logging
import unittest
import sys
import os
from itertools import product
import torch
os.environ['ENABLE_CNNL_TRYCATCH']='OFF' # pylint: disable=C0413
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase # pylint: disable=C0413
logging.basicConfig(level=logging.DEBUG)


class TestAddmmOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_contiguous(self):
        dtype_tensor_list = [torch.float, torch.float16]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((1, 234), (123, 648), (648, 234)),
                      ((123, 1), (123, 648), (648, 234)), ((0, 50), (0, 20), (20, 50)),
                      ((20), (10, 33), (33,20)), ((13), (4, 0), (0, 13)),
                      ((1, 1), (128, 1024), (1024, 379)), ((), (64, 128), (128, 256))]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype_tensor)
                m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
                m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
                res_mlu = torch.addmm(M_mlu, m1_mlu, m2_mlu)
                res_cpu = torch.addmm(M, m1, m2)
                self.assertTensorsEqual(res_cpu, res_mlu.cpu().float(), 0.003, use_MSE = True)

        # Test 0-strided
        for dtype_tensor in dtype_tensor_list:
            M = torch.randn((10, 1), dtype=torch.float).expand(10, 25)
            m1 = torch.randn((10, 1), dtype=torch.float).expand(10, 50)
            m2 = torch.randn((50, 25), dtype=torch.float)
            M_mlu = self.to_mlu_dtype(M, dtype_tensor)
            m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
            m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
            res_cpu = torch.addmm(M, m1, m2)
            res_mlu = torch.addmm(M_mlu, m1_mlu, m2_mlu)
            self.assertTensorsEqual(res_cpu, res_mlu.cpu().float(), 0.003, use_MSE = True)

        # Test alpha and beta not equal 1
        # TODO(guwei) temp fix for CNNLCORE-3321 # pylint: disable=W0511
        beta_list = [-0.5, 1.7, 1, 22]
        alpha_list = [-1.7, 0.4, 1, 33]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for beta in beta_list:
                    for alpha in alpha_list:
                        M = torch.randn(M_shape, dtype=torch.float)
                        m1 = torch.randn(m1_shape, dtype=torch.float)
                        m2 = torch.randn(m2_shape, dtype=torch.float)
                        M_mlu = self.to_mlu_dtype(M, dtype_tensor)
                        m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
                        m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
                        res_cpu = torch.addmm(input=M, beta=beta, mat1=m1,
                                              mat2=m2, alpha=alpha)
                        res_mlu = torch.addmm(input=M_mlu, beta=beta, mat1=m1_mlu,
                                              mat2=m2_mlu, alpha=alpha)
                        self.assertTensorsEqual(res_cpu, res_mlu.cpu().float(),
                                                0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_not_dense(self):
        dtype_tensor_list = [torch.float, torch.float16]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((1, 234), (123, 648), (648, 234)),
                      ((0, 50), (0, 20), (20, 50))]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn((m1_shape[0],2*m1_shape[-1]), dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype_tensor)[:, :int(M_shape[-1] / 2)]
                m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)[:, :m1_shape[-1]]
                m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)[:, :int(m2_shape[-1] / 2)]
                M = M[:, :int(M_shape[-1] / 2)]
                m1 = m1[:, :m1_shape[-1]]
                m2 = m2[:, :int(m2_shape[-1] / 2)]
                res_mlu = torch.addmm(M_mlu, m1_mlu, m2_mlu)
                res_cpu = torch.addmm(M, m1, m2)
                self.assertTensorsEqual(res_cpu, res_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_inplace_contiguous(self):
        dtype_tensor_list = [torch.float, torch.float16]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((2, 6), (2, 3), (3, 6)),
                      ((22, 58), (22, 45), (45, 58)), ((0, 50), (0, 20), (20, 50)),
                      ((4, 13), (4, 0), (0, 13))]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype_tensor)
                m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
                m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
                M_mlu.addmm_(m1_mlu, m2_mlu)
                M.addmm_(m1, m2)
                self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE = True)

        # Test 0-strided
        for dtype_tensor in dtype_tensor_list:
            M = torch.randn((10, 1), dtype=torch.float).expand(10, 25).clone()
            m1 = torch.randn((10, 1), dtype=torch.float).expand(10, 50).clone()
            m2 = torch.randn((50, 25), dtype=torch.float)
            M_mlu = self.to_mlu_dtype(M, dtype_tensor)
            m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
            m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
            M.addmm_(m1, m2)
            M_mlu.addmm_(m1_mlu, m2_mlu)
            self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE = True)

        # Test alpha and beta not equal 1
        # TODO(guwei) temp fix for CNNLCORE-3321 # pylint: disable=W0511
        beta_list = [-0.5, 1.7, 1, 22]
        alpha_list = [-1.7, 0.4, 1, 33]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for beta in beta_list:
                    for alpha in alpha_list:
                        M = torch.randn(M_shape, dtype=torch.float)
                        m1 = torch.randn(m1_shape, dtype=torch.float)
                        m2 = torch.randn(m2_shape, dtype=torch.float)
                        M_mlu = self.to_mlu_dtype(M, dtype_tensor)
                        m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
                        m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
                        M.addmm_(mat1=m1, mat2=m2, beta=beta, alpha=alpha)
                        M_mlu.addmm_(mat1=m1_mlu, mat2=m2_mlu, beta=beta, alpha=alpha)
                        self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_inplace_not_dense(self):
        dtype_tensor_list = [torch.float, torch.float16]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((22, 58), (22, 45), (45, 58)),
                      ((0, 50), (0, 20), (20, 50))]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn((m1_shape[0],2*m1_shape[-1]), dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype_tensor)[:, :int(M_shape[-1] / 2)]
                m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)[:, :m1_shape[-1]]
                m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)[:, :int(m2_shape[-1] / 2)]
                M = M[:, :int(M_shape[-1] / 2)]
                m1 = m1[:, :m1_shape[-1]]
                m2 = m2[:, :int(m2_shape[-1] / 2)]
                M_mlu.addmm_(m1_mlu, m2_mlu)
                M.addmm_(m1, m2)
                self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_out(self):
        dtype_tensor_list = [torch.float, torch.float16]
        shape_list = [((10, 25), (10, 50), (50, 25)), ((1, 234), (123, 648), (648, 234)),
                      ((123, 1), (123, 648), (648, 234)), ((0, 50), (0, 20), (20, 50)),
                      ((20,), (10, 33), (33,20)), ((13,), (4, 0), (0, 13)),
                      ((1, 1), (128, 1024), (1024, 379)), ((), (64, 128), (128, 256))]
        out_shapes = [(100, 10), (1,), (20, 20, 60, 100)]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for out_shape in out_shapes:
                    M = torch.randn(M_shape, dtype=torch.float)
                    m1 = torch.randn(m1_shape, dtype=torch.float)
                    m2 = torch.randn(m2_shape, dtype=torch.float)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), dtype_tensor)
                    M_mlu = self.to_mlu_dtype(M, dtype_tensor)
                    m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
                    m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
                    torch.addmm(M, m1, m2, out=out_cpu)
                    torch.addmm(M_mlu, m1_mlu, m2_mlu, out=out_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

        # test out equals input and out is slice
        for dtype_tensor in dtype_tensor_list:
            M = torch.randn((10, 25), dtype=torch.float)
            m1 = torch.randn((10, 50), dtype=torch.float)
            m2 = torch.randn((50, 25), dtype=torch.float)
            M_mlu = self.to_mlu_dtype(M, dtype_tensor)
            m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
            m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)

            torch.addmm(M, m1, m2, out=M[0:1, 0:1])
            torch.addmm(M_mlu, m1_mlu, m2_mlu, out=M_mlu[0:1, 0:1])
            self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE = True)

        # Test 0-strided
        for dtype_tensor in dtype_tensor_list:
            for out_shape in out_shapes:
                M = torch.randn((10, 1), dtype=torch.float).expand(10, 25)
                m1 = torch.randn((10, 1), dtype=torch.float).expand(10, 50)
                m2 = torch.randn((50, 25), dtype=torch.float)
                out_cpu = torch.randn(out_shape, dtype=torch.float)
                out_mlu = self.to_mlu_dtype(torch.randn(out_shape), dtype_tensor)
                M_mlu = self.to_mlu_dtype(M, dtype_tensor)
                m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
                m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
                torch.addmm(M, m1, m2, out=out_cpu)
                torch.addmm(M_mlu, m1_mlu, m2_mlu, out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True)

        # Test alpha and beta not equal 1
        # TODO(guwei) temp fix for CNNLCORE-3321 # pylint: disable=W0511
        beta_list = [-0.5, 1.7, 1, 22]
        alpha_list = [-1.7, 0.4, 1, 33]
        for dtype_tensor in dtype_tensor_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for beta, alpha, out_shape in product(beta_list, alpha_list, out_shapes):
                    M = torch.randn(M_shape, dtype=torch.float)
                    m1 = torch.randn(m1_shape, dtype=torch.float)
                    m2 = torch.randn(m2_shape, dtype=torch.float)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), dtype_tensor)
                    M_mlu = self.to_mlu_dtype(M, dtype_tensor)
                    m1_mlu = self.to_mlu_dtype(m1, dtype_tensor)
                    m2_mlu = self.to_mlu_dtype(m2, dtype_tensor)
                    torch.addmm(input=M, beta=beta, mat1=m1, mat2=m2,
                                alpha=alpha, out=out_cpu)
                    torch.addmm(input=M_mlu, beta=beta, mat1=m1_mlu, mat2=m2_mlu,
                                alpha=alpha, out=out_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                            0.003, use_MSE = True)
    #@unittest.skip("not test")
    @testinfo()
    def test_addmm_exception(self):
        M_mlu = torch.randn((10)).float().to('mlu')
        m1_mlu = torch.randn((10,21)).int().to('mlu')
        m2_mlu = torch.randn((21,10)).float().to('mlu')
        ref_msg = "matrix expected, 1D and 0 dimension tensor does not support inplace"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            M_mlu.addmm_(m1_mlu, m2_mlu)

        M_mlu = torch.randn(()).float().to('mlu')
        m1_mlu = torch.randn((10,21)).int().to('mlu')
        m2_mlu = torch.randn((21,10)).float().to('mlu')
        ref_msg = "matrix expected, 1D and 0 dimension tensor does not support inplace"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            M_mlu.addmm_(m1_mlu, m2_mlu)

        M_mlu = torch.randn((1,25)).float().to('mlu')
        m1_mlu = torch.randn((10,50)).float().to('mlu')
        m2_mlu = torch.randn((50,25)).float().to('mlu')
        ref_msg = "The expanded size of the tensor {} must match the ".format(m1_mlu.size(0))
        ref_msg += "existing size {} at non-singleton dimension 0.".format(M_mlu.size(0))
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            M_mlu.addmm_(m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,20)).float().to('mlu')
        m1_mlu = torch.randn((10,50)).float().to('mlu')
        m2_mlu = torch.randn((50,25)).float().to('mlu')
        ref_msg = "The expanded size of the tensor {} must match the ".format(m2_mlu.size(1))
        ref_msg += "existing size {} at non-singleton dimension 1.".format(M_mlu.size(1))
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            M_mlu.addmm_(m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).float().to('mlu')
        m1_mlu = torch.randn((10,50)).int().to('mlu')
        m2_mlu = torch.randn((50,25)).float().to('mlu')
        ref_msg = "Expected object of scalar type Float"
        ref_msg += " but got scalar type Int for argument #2 'mat1'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).float().to('mlu')
        m1_mlu = torch.randn((10,50)).float().to('mlu')
        m2_mlu = torch.randn((50,25)).int().to('mlu')
        ref_msg = "Expected object of scalar type Float"
        ref_msg += " but got scalar type Int for argument #3 'mat2'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).int().to('mlu')
        m1_mlu = torch.randn((10,50)).int().to('mlu')
        m2_mlu = torch.randn((50,25)).int().to('mlu')
        ref_msg = "addmm on mlu only support input tensors scalar type Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).float().to('mlu')
        m1_mlu = torch.randn((30,50)).float().to('mlu')
        m2_mlu = torch.randn((50,25)).float().to('mlu')
        ref_msg = "The expanded size of the tensor {} must match".format(m1_mlu.size(0))
        ref_msg += " the existing size {} at non-singleton dimension 0.".format(M_mlu.size(0))
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).float().to('mlu')
        m1_mlu = torch.randn((10,50)).float().to('mlu')
        m2_mlu = torch.randn((50,30)).float().to('mlu')
        ref_msg = "The expanded size of the tensor {} must match".format(m2_mlu.size(1))
        ref_msg += " the existing size {} at non-singleton dimension 1.".format(M_mlu.size(1))
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((30,10,25)).float().to('mlu')
        m1_mlu = torch.randn((30,10,50)).float().to('mlu')
        m2_mlu = torch.randn((30,50,25)).float().to('mlu')
        ref_msg = "The tensor of input's dim must 2, but found {}".format(M_mlu.dim())
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).float().to('mlu')
        m1_mlu = torch.randn((30,10,50)).float().to('mlu')
        m2_mlu = torch.randn((50,25)).float().to('mlu')
        ref_msg = "The tensor of mat1's dim must equal to input's dim, expected 2,"
        ref_msg += " but found {}".format(m1_mlu.dim())
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).float().to('mlu')
        m1_mlu = torch.randn((10,50)).float().to('mlu')
        m2_mlu = torch.randn((10,20,50,25)).float().to('mlu')
        ref_msg = "The tensor of mat2's dim must equal to input's dim, expected 2,"
        ref_msg += " but found {}".format(m2_mlu.dim())
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

        M_mlu = torch.randn((10,25)).float().to('mlu')
        m1_mlu = torch.randn((10,50)).float().to('mlu')
        m2_mlu = torch.randn((30,25)).float().to('mlu')
        ref_msg = "Size mismatch, the size of mat2's dim 0 must equal to the size of mat1's dim 1"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmm(M_mlu, m1_mlu, m2_mlu)

if __name__ == '__main__':
    unittest.main()
