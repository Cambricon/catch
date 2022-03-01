# pylint: disable=W0511
from __future__ import print_function

import unittest
import logging
import copy
from itertools import product
import sys
import os
import torch
from torch import nn

os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF'


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestAdaptivePoolingOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d(self):
        in_shape_list = [(8, 16, 14, 14), (16, 6, 8), (4, 23, 13, 64), (6, 8, 16), (4,64,128,128)]
        out_shape_list = [(4, 4), (10, 7), (9, 11), (2,2)]
        out_shape_list = [(2,2)]
        dtype_list = [torch.float, torch.half]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            if in_shape == (4,64,128,128) and dtype == torch.half:
                err = 0.03
            else:
                err = 3e-3
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool2d(out_shape)
            output_cpu = m(input_cpu)
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(),
                                    output_mlu.cpu().float(),
                                    err, use_MSE=True)
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    err, use_MSE=True)
            # test out
            output_mlu = self.to_device(torch.randn(output_cpu.size(), dtype=dtype))
            with torch.no_grad():
                torch._C._nn.adaptive_avg_pool2d(self.to_device(input_mlu), out_shape,
                  out=output_mlu)
                self.assertTensorsEqual(output_cpu.float(),
                                        output_mlu.cpu().float(),
                                        err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d(self):
        in_shape_list = [(8, 16, 14, 14), (16, 6, 8), (4, 23, 13, 64), (6, 8, 16)]
        out_shape_list = [(4, 4), (10, 7), (9, 11)]
        dtype_list = [torch.float, torch.half]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveMaxPool2d(out_shape)
            output_cpu = m(input_cpu.float())
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu.float(),
                                    output_mlu.cpu().float(),
                                    3e-3, use_MSE=True)
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)
            # test out
            output_mlu = self.to_device(torch.randn(output_cpu.size(), dtype=dtype))
            index_mlu = self.to_device(torch.randint(-10, 10, output_cpu.size()))
            with torch.no_grad():
                torch._C._nn.adaptive_max_pool2d(self.to_device(input_mlu), out_shape,
                  out=[output_mlu, index_mlu])
                self.assertTensorsEqual(output_cpu.float(),
                                        output_mlu.cpu().float(),
                                        3e-3, use_MSE=True)
        # test indices
        input_cpu = torch.arange(16).view(1, 4, 4).float()
        m = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)
        # Different with the origin CPU/GPU ops, the max indices returned by
        # MLU adaptive_max_pool2d_with_indices are local max indices inside the kernel
        output_cpu, _ = m(input_cpu)
        output_mlu, indices_mlu = m(self.to_device(input_cpu))
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
        self.assertTensorsEqual(torch.tensor([[[3, 3],[3, 3]]]), indices_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d_not_contiguous(self):
        in_shape_list = [(8, 16, 14, 14), (4, 23, 13, 64), (4,64,128,128)]
        out_shape_list = [(4, 4), (9, 11), (2, 2)]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[..., :-1]]
        list_list = [in_shape_list, out_shape_list, func_list, func_list]
        for in_shape, out_shape, in_func, out_func in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool2d(out_shape)
            output_cpu = m(in_func(input_cpu))
            output_mlu = m(in_func(self.to_device(input_mlu)))
            self.assertTensorsEqual(output_cpu.float(),
                                    output_mlu.cpu().float(),
                                    3e-3, use_MSE=True)
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad, retain_graph = True)
            output_mlu.backward(self.to_device(grad), retain_graph = True)
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)

            # test not dense grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1)
            output_cpu.backward(grad[..., :-1], retain_graph = True)
            output_mlu.backward(self.to_device(grad)[..., :-1], retain_graph = True)
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)

            # test channel last grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(self.convert_to_channel_last(grad))
            output_mlu.backward(self.convert_to_channel_last(self.to_device(grad)))
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)

            # test out
            output_mlu = out_func(self.to_device(torch.randn(output_cpu.size())))
            output_mlu_ptr = output_mlu.data_ptr()
            with torch.no_grad():
                torch._C._nn.adaptive_avg_pool2d(in_func(self.to_device(input_mlu)), out_shape,
                  out=output_mlu)
                self.assertTensorsEqual(output_cpu.float(),
                                        output_mlu.cpu().float(),
                                        3e-3, use_MSE=True)
                self.assertEqual(output_mlu_ptr, output_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d_not_contiguous(self):
        in_shape_list = [(8, 16, 14, 14), (4, 23, 13, 64)]
        out_shape_list = [(4, 4), (9, 11)]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[..., :-1]]
        list_list = [in_shape_list, out_shape_list, func_list, func_list]
        for in_shape, out_shape, in_func, out_func in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveMaxPool2d(out_shape)
            output_cpu = m(in_func(input_cpu.float()))
            output_mlu = m(in_func(self.to_device(input_mlu)))
            self.assertTensorsEqual(output_cpu.float(),
                                    output_mlu.cpu().float(),
                                    3e-3, use_MSE=True)
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad, retain_graph = True)
            output_mlu.backward(self.to_device(grad), retain_graph = True)
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)

            # test not dense grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1)
            output_cpu.backward(grad[..., :-1], retain_graph = True)
            output_mlu.backward(self.to_device(grad)[..., :-1], retain_graph = True)
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)

            # test channel last grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(self.convert_to_channel_last(grad))
            output_mlu.backward(self.convert_to_channel_last(self.to_device(grad)))
            self.assertTensorsEqual(input_cpu.grad.float(),
                                    input_mlu.grad.float(),
                                    3e-3, use_MSE=True)

            # test out
            output_mlu = out_func(self.to_device(torch.randn(output_cpu.size())))
            index_mlu = out_func(self.to_device(torch.randint(-10, 10, output_cpu.size())))
            output_mlu_ptr = output_mlu.data_ptr()
            index_mlu_ptr = index_mlu.data_ptr()
            with torch.no_grad():
                torch._C._nn.adaptive_max_pool2d(in_func(self.to_device(input_mlu)), out_shape,
                  out=[output_mlu, index_mlu])
                self.assertTensorsEqual(output_cpu.float(),
                                        output_mlu.cpu().float(),
                                        3e-3, use_MSE=True)
                self.assertEqual(output_mlu_ptr, output_mlu.data_ptr())
                self.assertEqual(index_mlu_ptr, index_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d_exception(self):
        input = torch.randn((2,3), dtype=torch.float).to('mlu')
        m = nn.AdaptiveAvgPool2d(7)
        ref_msg = r"^non-empty 3D or 4D \(batch mode\) tensor expected for input$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)  # pylint: disable=W0612

        input = torch.randn((2,3,0,7), dtype=torch.float).to('mlu')
        m = nn.AdaptiveAvgPool2d(7)
        ref_msg = r"^adaptive_avg_pool2d\(\): expected input to have"
        ref_msg = ref_msg + r" non-empty spatial dimensions, but input has sizes "
        ref_msg = ref_msg + r"\[2, 3, 0, 7\] with dimension 2 being empty$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

    #@unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d_exception(self):
        input = torch.randn((2,3), dtype=torch.float).to('mlu')
        m = nn.AdaptiveMaxPool2d(7)
        ref_msg = r"^non-empty 3D or 4D \(batch mode\) tensor expected for input$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)  # pylint: disable=W0612

        input = torch.randn((2,0,3,7), dtype=torch.float).to('mlu')
        m = nn.AdaptiveMaxPool2d(7)
        ref_msg = r"^adaptive_max_pool2d\(\): expected input to have"
        ref_msg = ref_msg + r" non-empty spatial dimensions, but input has sizes "
        ref_msg = ref_msg + r"\[2, 0, 3, 7\] with dimension 1 being empty$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

        input = torch.randn((2,2,3,7), dtype=torch.float).to('mlu')
        m = nn.AdaptiveMaxPool2d((2,2,2))
        ref_msg = r"^adaptive_max_pool2d: internal error: output_size\.size\(\) must be 2$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

if __name__ == '__main__':
    unittest.main()
