"""
test_div
"""
from __future__ import print_function

import unittest
import logging
import copy

import sys
import os
import itertools
import torch
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF'
import torch_mlu.core.mlu_model as ct  # pylint: disable=C0413,W0611

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD+"/../../")

from common_utils import testinfo, TestCase # pylint: disable=C0413,E0401,C0411

logging.basicConfig(level=logging.DEBUG)

class TestDivOp(TestCase):
    """
    test-div
    """
    #@unittest.skip("not test")
    @testinfo()
    def test_div_tensor_tensor(self):
        """
        test_tensor_tensor
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            # test_dim_0
            x_0 = torch.tensor(8.0)
            y_0 = torch.tensor(2.0)
            out_cpu = torch.div(x_0, y_0)
            out_mlu = torch.div(self.to_mlu_dtype(x_0, data_type),
                                self.to_mlu_dtype(y_0, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                    err, use_MSE=True)

            # need check again.
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1)),
                                   ((2, 30, 80), (2, 30, 80)),
                                   ((3, 20), (3, 20)),
                                   ((10,), (10,)),
                                   ((8732, 2), (8732, 2)),
                                   ((2, 2, 4, 2), (2,)),
                                   ((1, 2), (2, 2, 4, 2)),
                                   ((2, 1, 2, 4), (1, 2, 4)),
                                   ((1, 2, 4), (2, 1, 2, 4)),
                                   ((1, 3, 224, 224), (1, 1, 1, 1)),
                                   ((3, 3, 4, 224, 224), (4,224,224)),
                                   ((1, 3, 224), (1, 3, 1)),
                                   ((1, 3, 224, 224), (1,))]:
                for channel in channel_first:
                    x__ = torch.rand(shape1, dtype=torch.float)
                    y__ = torch.randint(low = 1, high = 10, size = shape2, dtype=torch.float)
                    y__ = y__ + 0.00005  # float range:[0.00005, 500]

                    input_check_x = x__.clone()
                    input_check_y = y__.clone()

                    out_cpu = torch.div(x__, y__)
                    #channel last test
                    if channel is False:
                        x__ = self.convert_to_channel_last(x__)
                    out_mlu = torch.div(self.to_mlu_dtype(x__, data_type),
                                        self.to_mlu_dtype(y__, torch.float))

                    self.assertTensorsEqual(input_check_x, x__, 0)
                    self.assertTensorsEqual(input_check_y, y__, 0)
                    # float type precision : 0.003
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float().contiguous(),
                                            err, use_MSE=True)

                for channel in channel_first:
                    x_cpu = torch.rand(shape1, dtype=data_type)
                    y_cpu = torch.randint(low = 1, high = 10, size = shape2, dtype=data_type)
                    y_cpu = y_cpu + 0.00005  # float range:[0.00005, 500]

                    #channel last test
                    if channel is False:
                        x_cpu = self.convert_to_channel_last(x_cpu)

                    x_mlu = copy.deepcopy(x_cpu).to("mlu")
                    y_mlu = copy.deepcopy(y_cpu).to("mlu")

                    tmp_cpu = torch.randn(1)
                    tmp_mlu = copy.deepcopy(tmp_cpu).to("mlu")
                    out_cpu = torch.div(x_cpu, y_cpu, out=tmp_cpu)
                    out_mlu = torch.div(x_mlu, y_mlu, out=tmp_mlu)

                    self.assertTensorsEqual(tmp_cpu, tmp_mlu.cpu(), err, use_MSE=True)
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float().contiguous(),
                                            err, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_div_not_contiguous_tensor_tensor(self):
        """
        test_tensor_tensor
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 10, 224, 224), (1, 10, 224, 1)),
                                   ((2, 30, 80), (2, 30, 80))]:
                x_ = torch.rand(shape1, dtype=torch.float)
                y_ = torch.randint(low = 1, high = 10, size = shape2, dtype=torch.float)
                y_ = y_ + 0.00005  # float range:[0.00005, 500]

                input_check_x = x_.clone()
                input_check_y = y_.clone()

                out_cpu = torch.div(x_[:,2:8,5:60], y_[:,2:8,5:60])
                out_mlu = torch.div(self.to_mlu_dtype(x_, data_type)[:,2:8,5:60],
                                    self.to_mlu_dtype(y_, torch.float)[:,2:8,5:60])

                self.assertTensorsEqual(input_check_x, x_, 0)
                self.assertTensorsEqual(input_check_y, y_, 0)
                # float type precision : 0.003
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float().contiguous(),
                                        err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_tensor_tensor_channel_last(self):
        """
        test_tensor_tensor
        """
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[..., ::2]]
        param_list = [dtype_list, func_list, func_list]
        #for data_type, err in dtype_list:
        for data_type, func_x, func_y in itertools.product(*param_list):
            for shape1, shape2 in [((224, 224), (1, 10, 224, 1)),
                                   ((1, 10, 224, 224), (1, 10, 224, 1))]:
                x_ = torch.rand(shape1, dtype=torch.float)
                y_ = torch.randint(low = 1, high = 10, size = shape2, dtype=torch.float)
                y_ = y_ + 0.00005  # float range:[0.00005, 500]

                x__ = x_
                y__ = y_

                input_check_x = x__.clone()
                input_check_y = y__.clone()

                out_cpu = torch.div(func_x(x__), func_y(y__))
                out_mlu = torch.div(func_x(self.to_mlu_dtype(x__, data_type)),
                                    func_y(self.to_mlu_dtype(y__, torch.float)))

                self.assertTensorsEqual(input_check_x, x__, 0)
                self.assertTensorsEqual(input_check_y, y__, 0)
                # float type precision : 0.003
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float().contiguous(),
                                        3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_inplace_not_contiguous_tensor_tensor(self):
        """
        test_tensor_tensor
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 10, 224, 224), (1, 10, 224, 1)),
                                   ((2, 30, 80), (1, 30, 80))]:
                x_ = torch.rand(shape1, dtype=torch.float)
                y_ = torch.randint(low = 1, high = 10, size = shape2, dtype=torch.float)
                y_ = y_ + 0.00005  # float range:[0.00005, 500]
                x__ = x_[:,2:8,5:60]
                y__ = y_[:,2:8,5:60]

                # inplace not contiguous div
                mlu_x__ = self.to_mlu_dtype(x__, data_type)
                mlu_y__ = self.to_mlu_dtype(y__, torch.float)
                mlu_x_dptr = mlu_x__.data_ptr()
                x__.div_(y__)
                mlu_x__.div_(mlu_y__)
                self.assertEqual(mlu_x_dptr, mlu_x__.data_ptr())
                self.assertTensorsEqual(x__, mlu_x__.cpu().float(),
                                        err, use_MSE=True)

        input = torch.randn(4, 6).fill_(2.)
        input_cpu = copy.deepcopy(input)
        input_mlu = copy.deepcopy(input_cpu).to('mlu')
        input_cpu[:, :4] /= 2.0
        input_mlu[:, :4] /= 2.0
        self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 3e-3, use_MSE=True)


    #@unittest.skip("not test")
    @testinfo()
    def test_div_inplace_tensor_tensor_channel_last(self):
        """
        test_tensor_tensor
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 10, 224, 224), (1, 10, 224, 1))]:
                x_ = torch.rand(shape1, dtype=torch.float)
                y_ = torch.randint(low = 1, high = 10, size = shape2, dtype=torch.float)
                y_ = y_ + 0.00005  # float range:[0.00005, 500]
                x__ = x_.to(memory_format=torch.channels_last)
                y__ = y_.to(memory_format=torch.channels_last)

                mlu_x__ = self.to_mlu_dtype(x__, data_type)
                mlu_y__ = self.to_mlu_dtype(y__, torch.float)
                mlu_x_dptr = mlu_x__.data_ptr()
                x__.div_(y__)
                mlu_x__.div_(mlu_y__)
                self.assertEqual(mlu_x_dptr, mlu_x__.data_ptr())
                self.assertTensorsEqual(x__, mlu_x__.cpu().float(),
                                        err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_tensor_scalar(self):
        """
        test_tensor_scalar
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # test_dim_0
            x_0 = torch.tensor(8.0)
            y_0 = 2.0
            out_cpu = torch.div(x_0, y_0)
            out_mlu = torch.div(self.to_mlu_dtype(x_0, data_type), y_0)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                    err, use_MSE=True)
            # test_input_0
            x_0 = torch.randn((0, 5))
            y_0 = 2.0
            out_cpu = torch.div(x_0, y_0)
            out_mlu = torch.div(self.to_mlu_dtype(x_0, data_type), y_0)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                    err, use_MSE=True)

            # test_input_0
            x_0 = torch.randint(low = 1, high = 10, size = (0, 5), dtype=torch.float)
            y_0 = 2.0
            out_cpu = torch.div(y_0, x_0)
            out_mlu = torch.div(y_0, self.to_mlu_dtype(x_0, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                    err, use_MSE=True)
            # test_input_0_inplace
            x_0 = torch.randn((0, 5))
            y_0 = 2.0
            x_mlu = self.to_mlu_dtype(x_0, data_type)
            x_0 /= y_0
            x_mlu /= y_0
            self.assertTensorsEqual(x_0, x_mlu.cpu().float(),
                                    err, use_MSE=True)
            shape_list = [(5), (0), (7, 9), (9, 8, 7), (1, 2, 3, 4),
                          (10, 10, 10, 10), (100, 200), (3, 40, 32),
                          (1111), (99, 30, 40), (34, 56, 78, 90),
                          (5, 6, 7, 8, 9), (9, 11, 12, 14, 15, 16)]
            # channel last test.
            channel_first = [True, False]
            for shape in shape_list:
                for channel in channel_first:
                    x__ = torch.rand(shape, dtype=torch.float)

                    input_check_x = x__.clone()

                    out_cpu_1 = torch.div(x__, 8.0)
                    if channel is False:
                        x__ = self.convert_to_channel_last(x__)
                    out_mlu_1 = torch.div(self.to_mlu_dtype(x__, data_type), 8.0)

                    self.assertTensorsEqual(input_check_x, x__, 0)
                    # float type precision : 0.003
                    self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu().float().contiguous(),
                                            err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_scalar_scalar(self):
        """
        test_scalar_scalar
        """
        shape_list = [(5), (7, 9), (9, 8, 7), (1, 2, 3, 4)]
        for shape in shape_list:
            x__ = torch.rand(shape, dtype=torch.float)

            input_check_x = x__.clone()

            out_cpu_2 = torch.div(x__.sum(), 8.0)
            out_mlu_2 = torch.div(self.to_mlu(x__).sum(), 8.0)

            self.assertEqual(out_cpu_2.dtype, out_mlu_2.dtype)
            self.assertTensorsEqual(input_check_x, x__, 0)
            # float type precision : 0.003
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(),
                                    3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_scalar_scalar_half(self):
        """
        test_scalar_scalar
        """
        shape_list = [(5), (7, 9), (9, 8, 7), (1, 2, 3, 4)]
        for shape in shape_list:
            x__ = torch.rand(shape, dtype=torch.float)

            input_check_x = x__.clone()

            out_cpu_2 = torch.div(x__.sum(), 8.0)
            out_mlu_2 = torch.div(self.to_mlu_dtype(x__, torch.half).sum(), 8.0)

            self.assertTensorsEqual(input_check_x, x__, 0)
            # float type precision : 0.003
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu().float(),
                                    3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_scalar_scalar_self(self):
        """
        test_scalar_scalar
        """
        shape_list = [(5), (7, 9), (9, 8, 7), (1, 2, 3, 4)]
        for shape in shape_list:
            x__ = torch.rand(shape, dtype=torch.float)
            out_origin = x__.sum()
            out_cpu_1 = out_origin.clone()
            out_cpu_1 /= 5.0
            out_mlu_1 = self.to_mlu(out_origin)
            out_mlu_ptr = out_mlu_1.data_ptr()
            out_mlu_1 /= 5.0
            self.assertEqual(out_mlu_ptr, out_mlu_1.data_ptr())
            self.assertEqual(out_cpu_1.dtype, out_mlu_1.dtype)

            out_cpu_2 = out_origin.clone()
            out_cpu_2 = out_cpu_2/5.0
            out_mlu_2 = self.to_mlu(out_origin)
            out_mlu_2 = out_mlu_2/5.0

            # float type precision : 0.003
            self.assertEqual(out_cpu_2.dtype, out_mlu_2.dtype)
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(),
                                    3e-3, use_MSE=True)
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(),
                                    3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_scalar_scalar_half_self(self):
        """
        test_scalar_scalar
        """
        shape_list = [(5), (7, 9), (9, 8, 7), (1, 2, 3, 4)]
        for shape in shape_list:
            x__ = torch.rand(shape, dtype=torch.float)
            out_origin = x__.sum()
            out_cpu_1 = out_origin.clone()
            out_cpu_1 /= 5.0
            out_mlu_1 = self.to_mlu_dtype(out_origin, torch.half)
            out_mlu_ptr = out_mlu_1.data_ptr()
            out_mlu_1 /= 5.0
            self.assertEqual(out_mlu_ptr, out_mlu_1.data_ptr())

            out_cpu_2 = out_origin.clone()
            out_cpu_2 = out_cpu_2/5.0
            out_mlu_2 = self.to_mlu_dtype(out_origin, torch.half)
            out_mlu_2 = out_mlu_2/5.0

            # float type precision : 0.003
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu().float(),
                                    3e-3, use_MSE=True)
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu().float(),
                                    3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_tensor_tensor_self(self):
        """
        test_tensor_tensor
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # test_dim_0
            x_0 = torch.tensor(8.0)
            y_0 = torch.tensor(2.0)
            out_origin = x_0
            out_cpu = out_origin.clone()
            out_cpu /= y_0
            out_mlu = self.to_mlu_dtype(out_origin, data_type)
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu /= self.to_mlu_dtype(y_0, data_type)
            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                    err, use_MSE=True)

            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1)),
                                   ((2, 30, 80), (2, 30, 80)),
                                   ((3, 20), (3, 20)),
                                   ((10), (10)),
                                   ((2, 1, 2, 4), (1, 2, 4)),
                                   ((1, 3, 224, 224), (1, 1, 1, 1)),
                                   ((1, 3, 224), (1, 3, 1)),
                                   ((1, 3, 224, 224), (1))]:
                x__ = torch.rand(shape1, dtype=torch.float)
                y__ = torch.rand(shape2, dtype=torch.float)
                y__ = y__ + 0.00005  # float range:[0.00005, 500]

                out_origin = x__
                out_cpu_1 = out_origin.clone()
                out_cpu_1 /= y__
                out_mlu_1 = self.to_mlu_dtype(out_origin, data_type)
                out_mlu_ptr = out_mlu_1.data_ptr()
                out_mlu_1 /= self.to_mlu_dtype(y__, data_type)
                self.assertEqual(out_mlu_ptr, out_mlu_1.data_ptr())

                out_cpu_2 = out_origin.clone()
                out_cpu_2 = out_cpu_2/y__
                out_mlu_2 = self.to_mlu_dtype(out_origin, data_type)
                out_mlu_2 = out_mlu_2/self.to_mlu_dtype(y__, data_type)

                # float type precision : 0.003
                self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu().float(),
                                        err, use_MSE=True)
                self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu().float(),
                                        err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_tensor_scalar_self(self):
        """
        test_tensor_scalar
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # test_dim_0
            x_0 = torch.tensor(8.0)
            y_0 = 2.0
            out_origin = x_0
            out_cpu = out_origin.clone()
            out_cpu /= y_0
            out_mlu = self.to_mlu_dtype(out_origin, data_type)
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu /= y_0
            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            if data_type == torch.float:
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(),
                                    err, use_MSE=True)

            shape_list = [(5), (7, 9), (9, 8, 7), (1, 2, 3, 4),
                          (10, 10, 10, 10), (100, 200), (3, 40, 32),
                          (1111), (99, 30, 40), (34, 56, 78, 90),
                          (5, 6, 7, 8, 9), (9, 11, 12, 14, 15, 16)]
            for shape in shape_list:
                x__ = torch.rand(shape, dtype=torch.float)

                out_origin = x__
                out_cpu_1 = out_origin.clone()
                out_cpu_1 /= 5.0
                out_mlu_1 = self.to_mlu_dtype(out_origin, data_type)
                out_mlu_ptr = out_mlu_1.data_ptr()
                out_mlu_1 /= 5.0
                self.assertEqual(out_mlu_ptr, out_mlu_1.data_ptr())

                out_cpu_2 = out_origin.clone()
                out_cpu_2 = out_cpu_2/5.0
                out_mlu_2 = self.to_mlu_dtype(out_origin, data_type)
                out_mlu_2 = out_mlu_2/5.0

                # float type precision : 0.003
                self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu().float(),
                                        err, use_MSE=True)
                self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu().float(),
                                        err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_tensor_scalar_with_different_datatype(self):
        """
        test_tensor_scalar
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        other_dtype_list = [torch.float, torch.half, torch.int, torch.short,
                            torch.long, torch.int8, torch.bool, torch.uint8]
        for data_type, err in dtype_list:
            shape_list = [(5,), (7, 9), (9, 8, 7), (1, 2, 3, 4),
                          (10, 10, 10, 10), (100, 200), (3, 40, 32),
                          (1111), (99, 30, 40), (34, 56, 78, 90),
                          (5, 6, 7, 8, 9), (9, 11, 12, 14, 15, 16)]
            for shape in shape_list:
                for other_data_type in other_dtype_list:
                    x__ = torch.rand(shape, dtype=torch.float)
                    y__ = torch.rand(shape, dtype=torch.float) + 1
                    y__ = y__.to(other_data_type)

                    out_origin = x__
                    out_cpu_1 = out_origin.clone()
                    out_cpu_1 /= y__
                    out_mlu_1 = self.to_mlu_dtype(out_origin, data_type)
                    out_mlu_ptr = out_mlu_1.data_ptr()
                    out_mlu_1 /= self.to_mlu_dtype(y__, other_data_type)
                    self.assertEqual(out_mlu_ptr, out_mlu_1.data_ptr())

                    mlu_2 = self.to_mlu_dtype(out_origin, data_type)
                    out_mlu_2 = mlu_2 / self.to_mlu_dtype(y__, other_data_type)

                    # float type precision : 0.003
                    if data_type == torch.float:
                        self.assertEqual(out_cpu_1.dtype, out_mlu_1.dtype)
                        self.assertEqual(out_cpu_1.dtype, out_mlu_2.dtype)
                    self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu().float(),
                                            err, use_MSE=True)
                    self.assertTensorsEqual(out_cpu_1, out_mlu_2.cpu().float(),
                                            err, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_div_exception(self):
        a = torch.randn(3) * 10
        b = torch.randn(3) * 10
        a_mlu = a.int().to('mlu')
        b_mlu = b.int().to('mlu')
        ref_msg = r"^Integer division of tensors using div or \/ is no longer supported, " \
                  + r"and in a future release div will perform true division as in Python 3\. " \
                  + r"Use true_divide or floor_divide \(\/\/ in Python\) instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.div(a_mlu, b_mlu)
        a = torch.randn((0, 1)).to('mlu')
        b = torch.randn((1, 0)).to('mlu')
        ref_msg = r"output with shape \[0, 1\] doesn't match the broadcast shape \[0, 0\]"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a /= b

if __name__ == '__main__':
    unittest.main()
