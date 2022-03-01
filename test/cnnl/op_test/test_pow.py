from __future__ import print_function

import sys
import os
import unittest
import logging
import itertools
import copy

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    @staticmethod
    def generate_input_data(value):
        if isinstance(value, tuple):
            # TODO:(liuwenhao)When the accuracy of cnnlpow is improved,
            # the input of pow can be negative
            return torch.rand(value, dtype=torch.float).abs() + 1
            # return torch.randn(value, dtype=torch.float)
        elif isinstance(value, int):
            return value
        elif isinstance(value, float):
            return value
        else:
            assert False, "Input type {0} not in [tuple,, int], is \
                           not support.".format(type(value))
            return None

    # @unittest.skip("not test")
    @testinfo()
    def test_pow(self):
        shape_list = [(2, 3, 4), (2, 3, 4, 3, 4, 2, 1), (2, 3, 4), (2, 3, 4, 3, 4, 2, 1), \
                      (2, 3, 4), 1, 3, 5]
        exp_list = [0.5, 2, 5, (2, 3, 4, 3, 4, 2, 1), (2, 3, 4), (2, 3, 4), \
                    (2, 3), (2, 3, 4, 3, 4, 2, 1)]
        data_types = [(torch.float, 3e-3), (torch.half, 3e-3)]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = self.generate_input_data(shape_list[i])
                exp1 = self.generate_input_data(exp_list[i])
                out_cpu = torch.pow(input1, exp1)
                out_mlu = torch.pow(self.to_mlu_dtype(input1, data_type), \
                                    self.to_mlu_dtype(exp1, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)
                if isinstance(exp1, int) and exp1 == 2 and data_type != torch.half:
                    input1 = torch.rand(shape_list[i], dtype=torch.float) * 10000
                    out_cpu = torch.pow(input1, exp1)
                    out_mlu = torch.pow(self.to_mlu_dtype(input1, data_type), exp1)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_pow_scalar(self):
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        for memory_format in memory_format_list:
            input_self = torch.rand(1, 3, 16, 16).to(memory_format=memory_format)
            exp1 = 3.2
            out_cpu = torch.pow(input_self, exp1)
            out_mlu = torch.pow(input_self.to("mlu"), exp1)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

            out_cpu = torch.pow(exp1, input_self)
            out_mlu = torch.pow(exp1, input_self.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

            input_mlu = copy.deepcopy(input_self)
            input_self.pow_(exp1)
            input_mlu = input_mlu.to("mlu")
            input_mlu.pow_(exp1)
            self.assertTensorsEqual(input_self, input_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_inplace(self):
        # input is not support scalar
        shape_list = [(2, 3, 4, 3, 4, 2, 1), (2, 3, 4), (2, 3, 4, 3, 4, 2, 1), (2, 3, 4)]
        exp_list = [2, 5, (2, 3, 4, 3, 4, 2, 1), (2, 3, 4)]
        data_types = [(torch.float, 3e-3), (torch.half, 3e-2)]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = self.generate_input_data(shape_list[i])
                exp1 = self.generate_input_data(exp_list[i])
                input1_mlu = self.to_mlu_dtype(input1, data_type)
                exp1_mlu = self.to_mlu_dtype(exp1, data_type)
                input1_ptr = input1_mlu.data_ptr()
                input1.pow_(exp1)
                input1_mlu.pow_(exp1_mlu)
                self.assertEqual(input1_ptr, input1_mlu.data_ptr())
                self.assertTensorsEqual(
                    input1.float(), input1_mlu.cpu().float(), err, use_MSE=True)
                if isinstance(exp1, int) and exp1 == 2 and data_type != torch.half:
                    input1 = torch.rand(shape_list[i], dtype=torch.float) * 10000
                    input1_mlu = self.to_mlu_dtype(input1, data_type)
                    input1_ptr = input1_mlu.data_ptr()
                    input1.pow_(exp1)
                    input1_mlu.pow_(exp1)
                    self.assertEqual(input1_ptr, input1_mlu.data_ptr())
                    self.assertTensorsEqual(
                        input1.float(), input1_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_channels_last(self):
        input_shape = (2, 3, 4, 3)
        other_shapes = [(2, 3, 4, 3), (1, 1, 1, 3), 3]
        for other_shape in other_shapes:
            input_cpu = self.generate_input_data(input_shape).to(memory_format = torch.channels_last)
            if isinstance(other_shape, int):
                exp_cpu = self.generate_input_data(other_shape)
            else:
                exp_cpu = self.generate_input_data(other_shape).to(memory_format = torch.channels_last)
            input_mlu = input_cpu.to('mlu')
            exp_mlu = self.to_mlu(exp_cpu)
            output_cpu = torch.pow(input_cpu, exp_cpu)
            output_mlu = torch.pow(input_mlu, exp_mlu)
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_memory_format_combination(self):
        input_shape = (2, 3, 4, 3)
        other_shapes = [(2, 3, 4, 3), (1, 1, 1, 3), 3]
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[:, :, :, :3]]
        param_list = [dtype_list, func_list, func_list]
        #for data_type, err in dtype_list:
        for data_type, func_x, func_y in itertools.product(*param_list):
            for other_shape in other_shapes:
                input_cpu = self.generate_input_data(input_shape)
                exp_cpu = self.generate_input_data(other_shape)

                input_mlu = input_cpu
                exp_mlu = exp_cpu

                if not isinstance(other_shape, int):
                    exp_cpu = func_y(exp_cpu)
                    exp_mlu = func_y(exp_mlu)
  
                out_cpu = torch.pow(func_x(input_cpu), exp_cpu)
                out_mlu = torch.pow(func_x(self.to_mlu_dtype(input_mlu, data_type)),
                                    self.to_mlu_dtype(exp_mlu, data_type))

                # float type precision : 0.003
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float().contiguous(),
                                        3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_inplace_channels_last(self):
        other_shape = (2, 3, 4, 3)
        input_shape = (2, 3, 4, 3)
        exp_cpu = self.generate_input_data(other_shape).to(memory_format = torch.channels_last)
        input_cpu = self.generate_input_data(input_shape).to(memory_format = torch.channels_last)
        input_mlu = self.to_mlu(input_cpu)
        exp_mlu = exp_cpu.to('mlu')
        input_ptr = input_mlu.data_ptr()
        input_cpu.pow_(exp_cpu)
        input_mlu.pow_(exp_mlu)

        self.assertEqual(input_ptr, input_mlu.data_ptr())
        self.assertTensorsEqual(
            input_cpu.float(), input_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_not_dense(self):
        other_shape = (1, 2, 2, 5)
        input_shape = (1, 2, 2, 5)
        exp_cpu = self.generate_input_data(other_shape)
        input_cpu = self.generate_input_data(input_shape)
        input_mlu = self.to_mlu(input_cpu)
        exp_mlu = exp_cpu.to('mlu')
        output_cpu = torch.pow(input_cpu[:, :, :, :3], exp_cpu[:, :, :, :3])
        output_mlu = torch.pow(input_mlu[:, :, :, :3], exp_mlu[:, :, :, :3])
        self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True)

        input_cpu[:, :, :, :3].pow_(exp_cpu[:, :, :, :3])
        input_mlu_ptr = input_mlu.data_ptr()
        input_mlu[:, :, :, :3].pow_(exp_mlu[:, :, :, :3])

        self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
        self.assertTensorsEqual(
            input_cpu.float(), input_mlu.cpu().float(), 3e-3, use_MSE=True)
 
if __name__ == "__main__":
    unittest.main()
