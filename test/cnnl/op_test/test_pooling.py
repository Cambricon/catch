from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import logging
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413 C0411

logging.basicConfig(level=logging.DEBUG)

class TestPoolingOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_avgpooling(self):
        # FIXME: Cnnl is not ready for ceil_mode=True, we will support it in the future. # pylint: disable=W0511
        shape_list = [(8, 16, 7, 7), (16, 6, 8, 16), (4, 23, 13, 64)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4]
        padding_v = [0, 1]
        ceil_mode_v = [False, False]
        include_pad_v = [False, True]

        loop_var = [
            shape_list, memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v,
            include_pad_v
        ]
        for in_shape, memory_format, kernel, stride, padding, ceil_mode, include_pad in product(
                *loop_var):
            input_t = torch.randn(in_shape, dtype=torch.float).to(memory_format = memory_format)
            avg_pool = nn.AvgPool2d(kernel,
                                    stride=stride,
                                    padding=padding,
                                    ceil_mode=ceil_mode,
                                    count_include_pad=include_pad)
            output_cpu = avg_pool(input_t)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)
            # not dense
            output_cpu = avg_pool(input_t[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu().float(),
                                    3e-3,
                                    use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpooling(self):
      # FIXME: Cnnl is not ready for ceil_mode=True, we will support it in the future.# pylint: disable=W0511
        in_shape = (4, 2, 128, 128)
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        input_t = torch.randn(in_shape, dtype=torch.float)
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]
        ceil_mode_v = [False, False]
        return_indices_v = [False]

        loop_var = [
            memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v, return_indices_v
        ]
        for memory_format, kernel, stride, padding, ceil_mode, return_indices in product(
                *loop_var):
            output_cpu = F.max_pool2d(input_t.to(memory_format = memory_format),
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)
            output_mlu = F.max_pool2d(self.to_mlu(input_t.to(memory_format = memory_format)),
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)

            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu(),
                                    3e-3,
                                    use_MSE=True)

            # not dense
            input_t = input_t.to(memory_format = memory_format)
            output_cpu = F.max_pool2d(input_t[:2, ...],
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)
            output_mlu = F.max_pool2d(self.to_mlu(input_t)[:2, ...],
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices,
                                      ceil_mode=ceil_mode)

            self.assertTensorsEqual(output_cpu,
                                    output_mlu.cpu(),
                                    3e-3,
                                    use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpooling_index(self):
      # FIXME: Cnnl is not ready for ceil_mode=True, we will support it in the future.# pylint: disable=W0511
        in_shape = (4, 2, 12, 12)
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        input_t = torch.randn(in_shape, dtype=torch.float)
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]
        ceil_mode_v = [False, False]
        return_indices_v = [True]

        loop_var = [memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v]
        for memory_format, kernel, stride, padding, ceil_mode in product(*loop_var):
            input_t = input_t.to(memory_format = memory_format)
            output_cpu = F.max_pool2d(input_t,
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices_v,
                                      ceil_mode=ceil_mode)

            output_mlu = F.max_pool2d(self.to_mlu(input_t),
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices_v,
                                      ceil_mode=ceil_mode)

            self.assertTensorsEqual(output_cpu[0],
                                    output_mlu[0].cpu(),
                                    3e-3,
                                    use_MSE=True)

            # not dense
            output_cpu = F.max_pool2d(input_t[:2, ...],
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices_v,
                                      ceil_mode=ceil_mode)

            output_mlu = F.max_pool2d(self.to_mlu(input_t)[:2, ...],
                                      kernel,
                                      stride=stride,
                                      padding=padding,
                                      dilation=1,
                                      return_indices=return_indices_v,
                                      ceil_mode=ceil_mode)

            self.assertTensorsEqual(output_cpu[0],
                                    output_mlu[0].cpu(),
                                    3e-3,
                                    use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpooling3d(self):
        shape_list = [(12, 2048, 1, 7, 7),
                      (12, 192, 8, 28, 28)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        data_types = [(torch.float, 3e-3), (torch.half, 3e-3)]
        kernel_v = [(1, 7, 7), (1, 3, 3)]
        stride_v = [(1, 1, 1), (1, 1, 1)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        include_pad_v = [True, True]

        loop_var = [
            shape_list, memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v,
            include_pad_v
        ]
        for in_shape, memory_format, kernel, stride, padding, ceil_mode, include_pad in zip(
                *loop_var):
            for data_type, err in data_types:
                if data_type == torch.float:
                    input_t = torch.randn(in_shape, dtype=torch.float).to(memory_format = memory_format)
                else:
                    input_t = torch.randn(in_shape, dtype=torch.half).to(dtype=torch.float, memory_format = memory_format)
                # test nn module
                avg_pool = nn.AvgPool3d(kernel,
                                        stride=stride,
                                        padding=padding,
                                        ceil_mode=ceil_mode,
                                        count_include_pad=include_pad)
                output_cpu = avg_pool(input_t)
                output_mlu = avg_pool(self.to_mlu_dtype(input_t, data_type))
                self.assertTensorsEqual(output_cpu,
                                        output_mlu.cpu().float(),
                                        3e-3,
                                        use_MSE=True)
                # test nn module not dense
                output_cpu = avg_pool(input_t[:2, ...])
                output_mlu = avg_pool(self.to_mlu_dtype(input_t, data_type)[:2, ...])
                self.assertTensorsEqual(output_cpu,
                                        output_mlu.cpu().float(),
                                        3e-3,
                                        use_MSE=True)
                # test function
                output_cpu = F.avg_pool3d(input_t,
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        ceil_mode=ceil_mode,
                                        count_include_pad=include_pad)
                output_mlu = F.avg_pool3d(self.to_mlu_dtype(input_t, data_type),
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        ceil_mode=ceil_mode,
                                        count_include_pad=include_pad)
                self.assertTensorsEqual(output_cpu,
                                        output_mlu.cpu().float(),
                                        3e-3,
                                        use_MSE=True)
                # test function not dense
                output_cpu = F.avg_pool3d(input_t[:2, ...],
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        ceil_mode=ceil_mode,
                                        count_include_pad=include_pad)
                output_mlu = F.avg_pool3d(self.to_mlu_dtype(input_t, data_type)[:2, ...],
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        ceil_mode=ceil_mode,
                                        count_include_pad=include_pad)
                self.assertTensorsEqual(output_cpu,
                                        output_mlu.cpu().float(),
                                        3e-3,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpooling3d(self):
        shape_list = [(12, 2048, 2, 7, 7),
                      (12, 128, 8, 112, 112)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(2, 1, 1), (2, 3, 3)]
        stride_v = [(1, 1, 1), (2, 2, 2)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        return_indices_v = [True, False]
        data_types = [(torch.float, 3e-3), (torch.half, 3e-3)]

        loop_var = [
            shape_list, memory_format_list, kernel_v, stride_v, padding_v, ceil_mode_v, return_indices_v
        ]
        for in_shape, memory_format, kernel, stride, padding, ceil_mode, return_indices in zip(
                *loop_var):
            for data_type, err in data_types:
                input_t = torch.randn(in_shape, dtype=torch.float).to(memory_format = memory_format)
                if data_type == torch.half:
                    input_t.half().float()
                # test nn module
                max_pool = nn.MaxPool3d(kernel,
                                        stride=stride,
                                        padding=padding,
                                        dilation=1,
                                        ceil_mode=ceil_mode,
                                        return_indices=return_indices)
                output_cpu = max_pool(input_t)
                output_mlu = max_pool(self.to_mlu_dtype(input_t, data_type))
                output_cpu_not_dense = max_pool(input_t[:2, ...])
                output_mlu_not_dense = max_pool(self.to_mlu_dtype(input_t, data_type)[:2, ...])
                if return_indices is True:
                    self.assertTensorsEqual(output_cpu[0],
                                            output_mlu[0].cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                    self.assertTensorsEqual(output_cpu_not_dense[0],
                                            output_mlu_not_dense[0].cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                else:
                    self.assertTensorsEqual(output_cpu,
                                            output_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                    self.assertTensorsEqual(output_cpu_not_dense,
                                            output_mlu_not_dense.cpu().float(),
                                            3e-3,
                                            use_MSE=True)

                # test function
                output_cpu = F.max_pool3d(input_t,
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        dilation=1,
                                        return_indices=return_indices,
                                        ceil_mode=ceil_mode)
                output_mlu = F.max_pool3d(self.to_mlu_dtype(input_t, data_type),
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        dilation=1,
                                        return_indices=return_indices,
                                        ceil_mode=ceil_mode)
                output_cpu_not_dense = F.max_pool3d(input_t[:2, ...],
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        dilation=1,
                                        return_indices=return_indices,
                                        ceil_mode=ceil_mode)
                output_mlu_not_dense = F.max_pool3d(self.to_mlu_dtype(input_t, data_type)[:2, ...],
                                        kernel,
                                        stride=stride,
                                        padding=padding,
                                        dilation=1,
                                        return_indices=return_indices,
                                        ceil_mode=ceil_mode)
                if return_indices is True:
                    self.assertTensorsEqual(output_cpu[0],
                                            output_mlu[0].cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                    self.assertTensorsEqual(output_cpu_not_dense[0],
                                            output_mlu_not_dense[0].cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                else:
                    self.assertTensorsEqual(output_cpu,
                                            output_mlu.cpu().float(),
                                            3e-3,
                                            use_MSE=True)
                    self.assertTensorsEqual(output_cpu_not_dense,
                                            output_mlu_not_dense.cpu().float(),
                                            3e-3,
                                            use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_maxpool2d_exception(self):
        input = torch.randn((2,3,8,8), dtype=torch.float).to('mlu')
        m = nn.MaxPool2d(kernel_size=(3,3,3), stride=2)
        m = m.to('mlu')
        ref_msg = r"^max_pool2d: kernel_size must either be a single int,"
        ref_msg = ref_msg + r" or a tuple of two ints$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2,3,8,8), dtype=torch.float).to('mlu')
        m = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2,2))
        m = m.to('mlu')
        ref_msg = r"^max_pool2d: stride must either be omitted,"
        ref_msg = ref_msg + r" a single int, or a tuple of two ints$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2,3,8), dtype=torch.float).to('mlu')
        m = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        m = m.to('mlu')
        ref_msg = r"^cnnl pool2d only support 4D input tensor currently\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2,3,8,8), dtype=torch.float).to('mlu')
        m = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), dilation=(1,2))
        m = m.to('mlu')
        ref_msg = r"^max_pool2d: dilation must be either a single int, or a tuple of two ints,"
        ref_msg = ref_msg + r" and cnnl pool2d only supports defalut dilation value$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

    #@unittest.skip("not test")
    @testinfo()
    def test_avgpool2d_exception(self):
        input = torch.randn((2,8,8), dtype=torch.float).to('mlu')
        m = nn.AvgPool2d(kernel_size=(3,3), stride=2)
        ref_msg = r"^cnnl pool2d only support 4D input tensor currently\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2,3,8,8), dtype=torch.float).to('mlu')
        m = nn.AvgPool2d(kernel_size=(3,3), stride=2, divisor_override=0)
        ref_msg = r"^divisor must be not zero$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

if __name__ == '__main__':
    unittest.main()
