from __future__ import print_function

import sys
import logging
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
from itertools import product
import unittest
import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestTypeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_type_param_empty(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (1000,), ()]
        dtype_list = [torch.half, torch.float,
                      torch.uint8, torch.int8, torch.short,
                      torch.int, torch.long, torch.bool]
        for shape, src_type in product(shape_list, dtype_list):
            if src_type in [torch.half, torch.float]:
                x = torch.randn(shape, dtype=src_type)
            elif src_type == torch.uint8:
                x = torch.randint(0, 255, shape).to(src_type)
            else:
                x = torch.randint(-128, 128, shape).to(src_type)
            out_cpu_type = x.type()
            out_mlu_type = x.to(ct.mlu_device()).type()
            l_tmp = out_cpu_type.split('.')
            l_tmp.insert(1, 'mlu')
            self.assertEqual('.'.join(l_tmp), out_mlu_type)

    # @unittest.skip("not test")
    @testinfo()
    def test_type_param_empty_channels_last(self):
        shape_list = [(512, 1024, 2, 2), (2, 3, 4, 5),
                      (254, 254, 112, 1), (2, 3, 24, 30), (1, 1, 1, 30)]
        dtype_list = [torch.half, torch.float,
                      torch.uint8, torch.int8, torch.short,
                      torch.int, torch.long, torch.bool]
        for shape, src_type in product(shape_list, dtype_list):
            if src_type in [torch.half, torch.float]:
                x = torch.randn(shape, dtype=src_type).to(memory_format = torch.channels_last)
            elif src_type == torch.uint8:
                x = torch.randint(0, 255, shape).to(src_type).to(
                    memory_format = torch.channels_last)
            else:
                x = torch.randint(-128, 128, shape).to(src_type).to(
                    memory_format = torch.channels_last)
            out_cpu_type = x.type()
            out_mlu_type = x.to(ct.mlu_device()).type()
            l_tmp = out_cpu_type.split('.')
            l_tmp.insert(1, 'mlu')
            self.assertEqual('.'.join(l_tmp), out_mlu_type)

    # @unittest.skip("not test")
    @testinfo()
    def test_type_param_empty_not_dense(self):
        shape_list = [(16, 32, 2, 30), (2, 3, 4, 32),
                      (24, 26, 112, 64), (2, 3, 24, 30), (1, 1, 1, 30)]
        dtype_list = [torch.half, torch.float,
                      torch.uint8, torch.int8, torch.short,
                      torch.int, torch.long, torch.bool]
        for shape, src_type in product(shape_list, dtype_list):
            if src_type in [torch.half, torch.float]:
                x = torch.randn(shape, dtype=src_type)[:, :, :, :15]
            elif src_type == torch.uint8:
                x = torch.randint(0, 255, shape).to(src_type)[:, :, :, :15]
            else:
                x = torch.randint(-128, 128, shape).to(src_type)[:, :, :, :15]
            out_cpu_type = x.type()
            out_mlu_type = x.to(ct.mlu_device()).type()
            l_tmp = out_cpu_type.split('.')
            l_tmp.insert(1, 'mlu')
            self.assertEqual('.'.join(l_tmp), out_mlu_type)

    # @unittest.skip("not test")
    @testinfo()
    def test_type_param_dtype(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4),
                      (254, 254, 112, 1, 1, 3), (1000,), ()]
        cast_map = {torch.float: {torch.half, torch.int, torch.short, torch.int8, torch.bool},
                    torch.half: {torch.float, torch.int, torch.short, torch.int8, torch.bool},
                    torch.long: {torch.float, torch.half, torch.short, torch.int8},
                    torch.int: {torch.float, torch.half, torch.short, torch.int8},
                    torch.short: {torch.float, torch.half, torch.int},
                    torch.int8: {torch.float, torch.half, torch.int},
                    torch.uint8: {torch.float, torch.half},
                    torch.bool: {torch.float, torch.half, torch.int},
                    }
        for shape, src_type in product(shape_list, cast_map.keys()):
            for dst_type in cast_map[src_type]:
                if src_type in [torch.half, torch.float]:
                    x = torch.randn(shape, dtype=src_type)
                elif src_type == torch.uint8:
                    x = torch.randint(0, 255, shape).to(src_type)
                else:
                    x = torch.randint(-128, 128, shape).to(src_type)
                for is_async in [False, True]:
                    out_cpu = x.type(dst_type, non_blocking=is_async)
                    out_mlu = x.to(ct.mlu_device()).type(dst_type, non_blocking=is_async)
                    self.assertEqual(out_mlu.dtype, dst_type)
                    self.assertEqual(out_cpu, out_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_type_param_dtype_channels_last(self):
        shape_list = [(512, 1024, 2, 2), (2, 3, 4, 16),
                      (254, 254, 112, 1), (2, 3, 24, 30), (1, 1, 1, 30)]
        cast_map = {torch.float: {torch.half, torch.int, torch.short, torch.int8, torch.bool},
                    torch.half: {torch.float, torch.int, torch.short, torch.int8, torch.bool},
                    torch.long: {torch.float, torch.half, torch.short, torch.int8},
                    torch.int: {torch.float, torch.half, torch.short, torch.int8},
                    torch.short: {torch.float, torch.half, torch.int},
                    torch.int8: {torch.float, torch.half, torch.int},
                    torch.uint8: {torch.float, torch.half},
                    torch.bool: {torch.float, torch.half, torch.int},
                    }
        for shape, src_type in product(shape_list, cast_map.keys()):
            for dst_type in cast_map[src_type]:
                if src_type in [torch.half, torch.float]:
                    x = torch.randn(shape, dtype=src_type).to(memory_format = torch.channels_last)
                elif src_type == torch.uint8:
                    x = torch.randint(0, 255, shape).to(src_type).to(
                        memory_format = torch.channels_last)
                else:
                    x = torch.randint(-128, 128, shape).to(src_type).to(
                        memory_format = torch.channels_last)
                for is_async in [False, True]:
                    out_cpu = x.type(dst_type, non_blocking=is_async)
                    out_mlu = x.to(ct.mlu_device()).type(dst_type, non_blocking=is_async)
                    self.assertEqual(out_mlu.dtype, dst_type)
                    self.assertEqual(out_cpu, out_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_type_param_dtype_not_dense(self):
        shape_list = [(16, 32, 2, 30), (2, 3, 4, 32),
                      (24, 26, 112, 64), (2, 3, 24, 30), (1, 1, 1, 30)]
        cast_map = {torch.float: {torch.half, torch.int, torch.short, torch.int8, torch.bool},
                    torch.half: {torch.float, torch.int, torch.short, torch.int8, torch.bool},
                    torch.long: {torch.float, torch.half, torch.short, torch.int8},
                    torch.int: {torch.float, torch.half, torch.short, torch.int8},
                    torch.short: {torch.float, torch.half, torch.int},
                    torch.int8: {torch.float, torch.half, torch.int},
                    torch.uint8: {torch.float, torch.half},
                    torch.bool: {torch.float, torch.half, torch.int},
                    }
        for shape, src_type in product(shape_list, cast_map.keys()):
            for dst_type in cast_map[src_type]:
                if src_type in [torch.half, torch.float]:
                    x = torch.randn(shape, dtype=src_type)[:, :, :, :15]
                elif src_type == torch.uint8:
                    x = torch.randint(0, 255, shape).to(src_type)[:, :, :, :15]
                else:
                    x = torch.randint(-128, 128, shape).to(src_type)[:, :, :, :15]
                for is_async in [False, True]:
                    out_cpu = x.type(dst_type, non_blocking=is_async)
                    out_mlu = x.to(ct.mlu_device()).type(dst_type, non_blocking=is_async)
                    self.assertEqual(out_mlu.dtype, dst_type)
                    self.assertEqual(out_cpu, out_mlu.cpu())

if __name__ == '__main__':
    unittest.main()
