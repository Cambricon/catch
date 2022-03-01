from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging
from itertools import product
import torch
import torch_mlu.core.mlu_model as ct       # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

class TestUniqueOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_unique(self):
        type_list = [
            torch.float, torch.int, torch.long, torch.double
        ]
        shape_list = [
            (64,), (4, 64), (3, 4, 64), (100, 64, 7, 7)
        ]
        # sorted == False is error and need to be fixed. just support True as gpu now
        sort_list = [True]
        inverse_list = [True, False]
        counts_list = [True, False]
        loop_var = [type_list, shape_list, sort_list, inverse_list, counts_list]
        for param in product(*loop_var):
            torch.manual_seed(1)
            t, shape, sort, inverse, counts = param
            input_cpu = torch.randint(0, 64, shape).to(t)
            input_mlu = input_cpu.to("mlu")

            if inverse and counts:
                output_cpu, inverse_indices_cpu, counts_cpu = torch.unique(input_cpu,
                        sorted=sort, return_inverse=inverse, return_counts=counts)
                output_mlu, inverse_indices_mlu, counts_mlu = torch.unique(input_mlu,
                        sorted=sort, return_inverse=inverse, return_counts=counts)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
                self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            elif inverse:
                output_cpu, inverse_indices_cpu = torch.unique(input_cpu,
                        sorted=sort, return_inverse=inverse)
                output_mlu, inverse_indices_mlu = torch.unique(input_mlu,
                        sorted=sort, return_inverse=inverse)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
            elif counts:
                output_cpu, counts_cpu = torch.unique(input_cpu, sorted=sort, return_counts=counts)
                output_mlu, counts_mlu = torch.unique(input_mlu, sorted=sort, return_counts=counts)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            else:
                output_cpu = torch.unique(input_cpu, sorted=sort)
                output_mlu = torch.unique(input_mlu, sorted=sort)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_channel_last(self):
        type_list = [torch.float, torch.int]
        shape_list = [(100, 64, 7, 7)]
        # sorted == False is error and need to be fixed. just support True as gpu now
        sort_list = [True]
        inverse_list = [True, False]
        counts_list = [True, False]
        loop_var = [type_list, shape_list, sort_list, inverse_list, counts_list]
        for param in product(*loop_var):
            torch.manual_seed(1)
            t, shape, sort, inverse, counts = param
            input_cpu = torch.randint(0, 64, shape).to(t).to(memory_format=torch.channels_last)
            input_mlu = input_cpu.to("mlu")

            if inverse and counts:
                output_cpu, inverse_indices_cpu, counts_cpu = torch.unique(input_cpu,
                        sorted=sort, return_inverse=inverse, return_counts=counts)
                output_mlu, inverse_indices_mlu, counts_mlu = torch.unique(input_mlu,
                        sorted=sort, return_inverse=inverse, return_counts=counts)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
                self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            elif inverse:
                output_cpu, inverse_indices_cpu = torch.unique(input_cpu,
                        sorted=sort, return_inverse=inverse)
                output_mlu, inverse_indices_mlu = torch.unique(input_mlu,
                        sorted=sort, return_inverse=inverse)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
            elif counts:
                output_cpu, counts_cpu = torch.unique(input_cpu, sorted=sort, return_counts=counts)
                output_mlu, counts_mlu = torch.unique(input_mlu, sorted=sort, return_counts=counts)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            else:
                output_cpu = torch.unique(input_cpu, sorted=sort)
                output_mlu = torch.unique(input_mlu, sorted=sort)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_not_dense(self):
        type_list = [
            torch.float, torch.int
        ]
        shape_list = [
            (64,), (4, 64), (3, 4, 64), (100, 64, 7, 7)
        ]
        # sorted == False is error and need to be fixed. just support True as gpu now
        sort_list = [True]
        inverse_list = [True, False]
        counts_list = [True, False]
        loop_var = [type_list, shape_list, sort_list, inverse_list, counts_list]
        for param in product(*loop_var):
            torch.manual_seed(1)
            t, shape, sort, inverse, counts = param
            input_cpu = torch.randint(0, 64, shape).to(t)
            input_mlu = input_cpu.to("mlu")

            if inverse and counts:
                output_cpu, inverse_indices_cpu, counts_cpu = torch.unique(input_cpu[::2],
                        sorted=sort, return_inverse=inverse, return_counts=counts)
                output_mlu, inverse_indices_mlu, counts_mlu = torch.unique(input_mlu[::2],
                        sorted=sort, return_inverse=inverse, return_counts=counts)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
                self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            elif inverse:
                output_cpu, inverse_indices_cpu = torch.unique(input_cpu[::2],
                        sorted=sort, return_inverse=inverse)
                output_mlu, inverse_indices_mlu = torch.unique(input_mlu[::2],
                        sorted=sort, return_inverse=inverse)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
            elif counts:
                output_cpu, counts_cpu = torch.unique(input_cpu[::2],
                        sorted=sort, return_counts=counts)
                output_mlu, counts_mlu = torch.unique(input_mlu[::2],
                        sorted=sort, return_counts=counts)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
                self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            else:
                output_cpu = torch.unique(input_cpu[::2], sorted=sort)
                output_mlu = torch.unique(input_mlu[::2], sorted=sort)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_exception(self):
        x_mlu = torch.arange(1, 9, dtype=torch.uint8).to("mlu")
        ref_msg = "Expected tensor for argument #1 'input' to have one of the"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.unique(x_mlu, sorted=True, return_inverse=True, return_counts=True)

if __name__ == '__main__':
    unittest.main()
