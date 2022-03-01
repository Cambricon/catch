from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_contiguous(self):
        # torch.manual_seed(0)
        shape_list = [(10,), (2, 2, 3), (2, 0, 3), (2, 3, 4, 5)]
        dtype_list = [torch.bool, torch.float32, torch.int32, torch.double, torch.long]
        for dtype in dtype_list:
            for shape in shape_list:
                a = torch.randint(3, shape).type(dtype)
                result_cpu = torch.nonzero(a, as_tuple=False)
                result_mlu = torch.nonzero(self.to_device(a), as_tuple=False)
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            ## test scalar input
            a = torch.tensor(0).type(dtype)
            result_cpu = torch.nonzero(a, as_tuple=False)
            result_mlu = torch.nonzero(self.to_device(a), as_tuple=False)
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_channel_last(self):
        # torch.manual_seed(0)
        shape = (2, 3, 4, 5)
        dtype_list = [torch.bool, torch.float32, torch.int32, torch.double, torch.long]
        for dtype in dtype_list:
            a = torch.randint(3, shape).type(dtype)
            a = self.convert_to_channel_last(a)
            result_cpu = torch.nonzero(a, as_tuple=False)
            result_mlu = torch.nonzero(self.to_device(a), as_tuple=False)
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_not_dense(self):
        # torch.manual_seed(0)
        shape_list = [(2, 2, 6), (2, 3, 4, 10)]
        dtype_list = [torch.bool, torch.float32, torch.int32, torch.double, torch.long]
        for dtype in dtype_list:
            for shape in shape_list:
                a = torch.empty(0)
                if len(shape_list) == 3:
                    a = torch.randint(3, shape).type(dtype)[:, :, :int(shape[-1] / 2)]
                elif len(shape_list) ==4:
                    a = torch.randint(3, shape).type(dtype)[:, :, :, :int(shape[-1] / 2)]
                result_cpu = torch.nonzero(a, as_tuple=False)
                result_mlu = torch.nonzero(self.to_device(a), as_tuple=False)
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_out_contiguous(self):
        #torch.manual_seed(0)
        a = torch.randint(3, (2, 2, 3)).type(torch.bool)
        # the element number of out >= the expected of the op
        out_cpu = torch.randint(3, (a.numel() * a.dim(),))
        out_mlu = self.to_device(torch.randint(3, (a.numel() * a.dim(),)))
        origin_ptr = out_mlu.data_ptr()
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
        self.assertEqual(origin_ptr, out_mlu.data_ptr())
        # the element number of out < the expected of the op
        out_cpu = torch.randint(3, (1, ))
        out_mlu = self.to_device(torch.randint(3, (1, )))
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

        ## test zero elements
        a = torch.randint(3, (2, 0, 3)).type(torch.bool)
        out_cpu = torch.randint(3, (a.numel() * a.dim(),))
        out_mlu = self.to_device(torch.randint(3, (a.numel() * a.dim(),)))
        origin_ptr = out_mlu.data_ptr()
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
        self.assertEqual(origin_ptr, out_mlu.data_ptr())

        ## test scalar input
        a = torch.tensor(1).type(torch.bool)
        out_cpu = torch.randint(3, (1,))
        out_mlu = self.to_device(torch.randint(3, (1,)))
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_out_channel_last(self):
        #torch.manual_seed(0)
        a = torch.randint(3, (2, 2, 3, 4)).type(torch.bool)
        a = self.convert_to_channel_last(a)
        # the element number of out >= the expected of the op
        out_cpu = torch.randint(3, (a.numel() * a.dim(),))
        out_mlu = self.to_device(torch.randint(3, (a.numel() * a.dim(),)))
        origin_ptr = out_mlu.data_ptr()
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
        self.assertEqual(origin_ptr, out_mlu.data_ptr())
        # the element number of out < the expected of the op
        out_cpu = torch.randint(3, (1, ))
        out_mlu = self.to_device(torch.randint(3, (1, )))
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_out_not_dense(self):
        #torch.manual_seed(0)
        a = torch.randint(3, (2, 2, 3, 4)).type(torch.bool)[:, :, :, :2]
        # the element number of out >= the expected of the op
        out_cpu = torch.randint(3, (a.numel() * a.dim(),))
        out_mlu = self.to_device(torch.randint(3, (a.numel() * a.dim(),)))
        origin_ptr = out_mlu.data_ptr()
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
        self.assertEqual(origin_ptr, out_mlu.data_ptr())
        # the element number of out < the expected of the op
        out_cpu = torch.randint(3, (1, ))
        out_mlu = self.to_device(torch.randint(3, (1, )))
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_as_tuple_contiguous(self):
        # torch.manual_seed(0)
        shape_list = [(2, 2, 3)]
        for shape in shape_list:
            a = torch.randint(3, shape).type(torch.bool)
            result_cpu = torch.nonzero(a, as_tuple=True)
            result_mlu = torch.nonzero(self.to_device(a), as_tuple=True)
            self.assertEqual(len(result_mlu), len(result_cpu))
            for out_cpu, out_mlu in zip(result_cpu, result_mlu):
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_as_tuple_channel_last(self):
        # torch.manual_seed(0)
        shape_list = [(2, 2, 3, 8)]
        for shape in shape_list:
            a = torch.randint(3, shape).type(torch.bool)
            a = self.convert_to_channel_last(a)
            result_cpu = torch.nonzero(a, as_tuple=True)
            result_mlu = torch.nonzero(self.to_device(a), as_tuple=True)
            self.assertEqual(len(result_mlu), len(result_cpu))
            for out_cpu, out_mlu in zip(result_cpu, result_mlu):
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_as_tuple_not_dense(self):
        # torch.manual_seed(0)
        shape_list = [(2, 2, 6)]
        for shape in shape_list:
            a = torch.randint(3, shape).type(torch.bool)[:, :, :int(shape[-1] / 2)]
            result_cpu = torch.nonzero(a, as_tuple=True)
            result_mlu = torch.nonzero(self.to_device(a), as_tuple=True)
            self.assertEqual(len(result_mlu), len(result_cpu))
            for out_cpu, out_mlu in zip(result_cpu, result_mlu):
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_nonzero_expetion(self):
        shape = (2, 2, 3)
        a = torch.randint(3, shape).type(torch.int8).to('mlu')
        ref_msg = "self dtype of mlu nonzero op not implemented for Char"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.nonzero(a, as_tuple=True)

if __name__ == "__main__":
    unittest.main()
