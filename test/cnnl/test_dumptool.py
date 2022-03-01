from __future__ import print_function
# pylint: disable=C0411, C0412, C0413, C0415, W0612, R0201, W0223
import sys
import logging
import os
import shutil
import torch
from torch import nn
from itertools import product
import unittest
os.environ['ENABLE_CNNL_TRYCATCH']='OFF'
from torch_mlu.core.dumptool import Dumper
from torch_mlu.core.dumptool import dump_cnnl_gencase
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
logging.basicConfig(level=logging.DEBUG)

def checkfile(path, data):
    assert os.path.isfile(path), f"Cannot find dumpdir {path}"
    with open(path, "r") as f:
        st = f.readlines()
    assert len(st) == len(data),\
           f"Expect to Dump {len(data)} lines in {path}, but {len(st)} lines found. "\
           + f"The result should be {data}"
    for i, (a, b) in enumerate(zip(st, data)):
        # -1 to skip \n at last
        assert a[:-1] == str(b) or b is None,\
               f"Line {i} should be '{b}', but '{a}' found here."

class TestDumptool(TestCase):

    #@unittest.skip("not test")
    @testinfo()
    def test_tensor_dump(self):
        '''
        Test tensor dump by a view of float tensor, compare with a view of int tensor.
        Thus we can test dump tensor with strides, dtypes and dump levels.
        '''
        # use .1 instead .0 to make sure dump result won't ellipsis '.0'
        data_float = [-2.1, -1.1, 0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1]
        data_int = [[-1, -2, -3],
                    [2, 3, 4],
                    [-3, -3, -3],
                    [-4, 3, 4],
                    [5, -3, 4],
                    [6, 3, -4],
                    [7, 5, 6],
                    [8, 0, 0],
                    [-9, -9, -9],
                    [-10, -9, -9],
                    [-11, -9, -9],
                    [12, 9, 9]]

        float_result = [data_float[:10], [49.2], data_float]  # check 1
        int_input = [_[0] for _ in data_int]
        int_result = [int_input[:10], [78], int_input]  # check 2
        # bool expect is result of data_int[:][0] > 0
        bool_expect = [int(_ > 0) for _ in int_input]
        bool_result = [bool_expect[:10], [6], bool_expect]  # check 3
        dump_number = [10, 1, 12]

        for enable, use_cpu, level in product([True, False], [True, False], [0,1,2]):
            dir_name = "Dump" + str(enable) + str(use_cpu) + str(level)
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
            tensor1 = torch.tensor([data_float, data_float], dtype=torch.float).to('mlu')
            tensor2 = torch.tensor(data_int, dtype=torch.int).to('mlu')
            with Dumper(dir_name, enable, use_cpu, level):
                _ = tensor1[0]
                _ = tensor2[:, 0] > 0
            if not enable:
                assert os.path.isdir(dir_name) is False, "Dumper is not Enabled but dir founds!"
                continue
            float_info = "Tensor : Type = Float : Shape = [12] : Stride = [1]"\
                        + f" : Dumped = {dump_number[level]}"
            int_info = "Tensor : Type = Int : Shape = [12] : Stride = [3]"\
                        + f" : Dumped = {dump_number[level]}"
            bool_info = "Tensor : Type = Bool : Shape = [12] : Stride = [1]"\
                        + f" : Dumped = {dump_number[level]}"
            # Check the float tensor, int tensor and bool tensor from different op data
            checkfile(f"{dir_name}/1_select/cnnl_result",
                      [float_info] + float_result[level])
            checkfile(f"{dir_name}/4_gt/mlu_self",
                      [int_info] + int_result[level])
            checkfile(f"{dir_name}/4_gt/cnnl_result",
                      [bool_info] + bool_result[level])

    @unittest.skip("not test")
    @testinfo()
    def test_generator_dump(self):
        x = torch.tensor([1,2,3,4], dtype=torch.float).to('mlu')
        g = torch.Generator()
        if os.path.isdir("DumpGenerator"):
            shutil.rmtree("DumpGenerator")
        with Dumper("DumpGenerator", True, False, 0):
            x.bernoulli_(0.1, generator=g)
        checkfile("./DumpGenerator/1_bernoulli_/mlu_generator",
                  ["Optional : ",
                   "Generator : ",
                   None,  # None means a random data and no need to compare
                  'cpu'])

    # @unittest.skip("not test")
    @testinfo()
    def test_tuple_dump(self):
        x = torch.tensor([1,2,3,4], dtype=torch.float).to('mlu')
        if os.path.isdir("DumpTuple"):
            shutil.rmtree("DumpTuple")
        with Dumper("DumpTuple", True, False, 0):
            u,v = torch.max(x, dim=0, keepdim=True)
        checkfile("./DumpTuple/1_max/cnnl_result",
                  ["Tuple : Size = 2",
                   "Tensor : Type = Float : Shape = [1] : Stride = [1] : Dumped = 1",
                   4,
                   "Tensor : Type = Long : Shape = [1] : Stride = [1] : Dumped = 1",
                   3])

    # @unittest.skip("not test")
    @testinfo()
    def test_array_dump(self):
        x = torch.randn(1,4,4,4, dtype=torch.float).to('mlu')
        if os.path.isdir("DumpArray"):
            shutil.rmtree("DumpArray")
        weight = torch.randn(4, dtype=torch.float)
        class BN(nn.Module):
            def __init__(self):
                super(BN, self).__init__()
                self.features = nn.BatchNorm2d(4, affine=True)
                self.features.weight = nn.Parameter(weight)
            def forward(self, x):
                output = self.features(x)
                return output
        grad = torch.randn(1,4,4,4, dtype=torch.float).to('mlu')
        layer = BN().train().float().to('mlu')
        with Dumper("DumpArray", True, False, 0):
            out = layer(x)
            out.backward(grad)
        checkfile("./DumpArray/3_native_batch_norm_backward/mlu_output_mask",
                  ["Array : Size = 3",
                   0,
                   1,
                   1])

    # @unittest.skip("not test")
    @testinfo()
    def test_dump_cnnl_gencase(self):
        a = torch.randn((2,2,6,6))
        b = torch.randn((2,2,6,6))

        gencase_files = './gen_case'
        for level in ['L1', 'L2']:
            if os.path.exists(gencase_files):
                shutil.rmtree(gencase_files)
                print("Remove gencase files: ", gencase_files)
            dump_cnnl_gencase(enable=True, level=level)
            out = (a.to('mlu') + b.to('mlu')).view(4, 36)
            dump_cnnl_gencase(enable=False)
            self.assertTrue(os.path.exists(gencase_files))

        for level in ['L3']:
            dump_cnnl_gencase(enable=True, level=level)
            out = (a.to('mlu') + b.to('mlu')).view(4, 36)
            dump_cnnl_gencase(enable=False)

    # @unittest.skip("not test")
    @testinfo()
    def test_dump_cnnl_gencase_invalid_use(self):
        a = torch.randn((2,2))
        b = torch.randn((2,2))
        for level in ['L0', 'l1', 'L4']:
            with self.assertRaises(ValueError):
                dump_cnnl_gencase(enable=True, level=level)
                out = a.to('mlu') + b.to('mlu')
                dump_cnnl_gencase(enable=False)

if __name__ == '__main__':
    unittest.main()
