from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=all
import unittest
import logging
import copy
import random as rd

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestBitwiseOrOps(TestCase):
    def _generate_tensor(self, shape, dtype):
        if dtype == torch.bool:
            out = torch.randint(2, shape).type(dtype)
        elif dtype == torch.uint8:
            out = torch.randint(16777216, shape).type(dtype)
        else:
            out = torch.randint(-16777216, 16777216, shape).type(dtype)
        return out

    def _generate_scalar(self, dtype):
        if dtype == torch.bool:
            out = rd.choice([True, False])
        elif dtype == torch.uint8:
            out = rd.randint(0, 16777216)
        else:
            out = rd.randint(-16777216, 16777216)
        return out

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_or(self):
        dtype_lst = [(torch.bool, torch.bool),
                      (torch.uint8, torch.uint8),
                      (torch.int16, torch.int16),
                      (torch.int32, torch.int32),
                      (torch.long, torch.long),
                      (torch.int32, torch.int8),
                      (torch.int32, torch.int16),
                      (torch.int8, torch.int8)]
        for dtype1, dtype2 in dtype_lst:
            for shape1, shape2 in [((1, 2,), (1, 1)),
                                   ((2, 30, 80), (2, 30, 80)),
                                   ((3, 20), (3, 20)),
                                   ((3, 273), (1, 273)),
                                   ((1, 273), (3, 273)),
                                   ((2, 2, 4, 2), (1, 2)),
                                   ((1, 2), (2, 2, 4, 2)),
                                   ((1, 3, 224, 224), (1, 1, 1)),
                                   ((1, 1, 1), (1, 3, 224, 224)),
                                   ((1, 1, 3), (1, 224, 3)),
                                   ((1, 3, 224), (1, 3, 1)),
                                   ((1, 3, 1), (1, 3, 224)),
                                   ((1, 3, 224, 224), (1, 1))]:
                input1 = self._generate_tensor(shape1, dtype1)
                input2 = self._generate_tensor(shape2, dtype2)
                result_cpu = input1 | input2
                result_mlu = self.to_device(input1) | self.to_device(input2)
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

                # test no dense
                x = self._generate_tensor(shape1, dtype1)[...,:2]
                y = self._generate_tensor(shape2, dtype2)[...,:2]
                x_mlu = self.to_device(copy.deepcopy(x))
                y_mlu = self.to_device(copy.deepcopy(y))
                out_cpu = x.bitwise_or(y)
                out_mlu = x_mlu.bitwise_or(y_mlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003,
                                        use_MSE = True)

                # test scalar input
                # FIXME(liuyuxin): scalar computation are not supported for some types
                # will be fixed in future.
                if dtype1 == torch.uint8 or dtype1 == torch.int16 or dtype1 == torch.int8:
                   continue
                input3 = self._generate_scalar(dtype2)
                result_cpu = input1 | input3
                result_mlu = self.to_device(input1) | input3
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)


                if input1.dim() == 4:
                    # test channels last
                    x = self._generate_tensor(shape1, dtype1).to(memory_format = torch.channels_last)
                    y = self._generate_tensor(shape2, dtype2)
                    x_mlu = self.to_device(copy.deepcopy(x))
                    y_mlu = self.to_device(copy.deepcopy(y))
                    out_cpu = x | y
                    out_mlu = x_mlu | y_mlu
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003,
                                            use_MSE = True)

        input1 = self._generate_tensor((1, 20, 30, 40), torch.int)
        input3 = self._generate_scalar(torch.int)
        out_big = self._generate_tensor((5, 20, 30, 40), dtype=torch.int)
        out_big_mlu = self.to_device(out_big)
        torch.bitwise_or(input1, input3, out=out_big)
        torch.bitwise_or(self.to_device(input1), self.to_device(input3), out = out_big_mlu)
        self.assertTensorsEqual(out_big.float(), out_big_mlu.cpu().float(), 0, use_MSE = True)

        input1 = self._generate_tensor((1, 20, 30, 40), torch.int)
        input3 = self._generate_tensor((4, 20, 30, 40), torch.int)
        out_big = self._generate_tensor((5, 20, 30, 40), dtype=torch.int)
        out_big_mlu = self.to_device(out_big)
        torch.bitwise_or(input1, input3, out=out_big)
        torch.bitwise_or(self.to_device(input1), self.to_device(input3), out = out_big_mlu)
        self.assertTensorsEqual(out_big.float(), out_big_mlu.cpu().float(), 0, use_MSE = True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_or_inplace(self):
        dtype_lst = [(torch.bool, torch.bool),
                      (torch.uint8, torch.uint8),
                      (torch.int16, torch.int16),
                      (torch.int32, torch.int32),
                      (torch.long, torch.long),
                      (torch.int32, torch.int8),
                      (torch.int32, torch.int16),
                      (torch.int8, torch.int8)]
        for dtype1, dtype2 in dtype_lst:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1)),
                                   ((2, 30, 80), (2, 30, 80)),
                                   ((3, 20), (3, 20)),
                                   ((3, 273), (1, 273)),
                                   ((2, 2, 4, 2), (1, 2)),
                                   ((1, 3, 224, 224), (1, 1, 1)),
                                   ((1, 3, 224), (1, 3, 1)),
                                   ((1, 3, 224, 224), (1, 1))]:
                input1 = self._generate_tensor(shape1, dtype1)
                input2 = self._generate_tensor(shape2, dtype2)
                input1_copy = copy.deepcopy(input1)
                input1.bitwise_or_(input2)
                input1_mlu = self.to_device(input1_copy)
                raw_ptr = input1_mlu.data_ptr()
                input1_mlu.bitwise_or_(self.to_device(input2))
                cur_ptr = input1_mlu.data_ptr()
                self.assertEqual(raw_ptr, cur_ptr)
                self.assertTensorsEqual(input1, input1_mlu.cpu(), 0)

                # test no dense
                x = self._generate_tensor(shape1, dtype1)[...,:2]
                y = self._generate_tensor(shape2, dtype2)[...,:2]
                x_mlu = self.to_device(copy.deepcopy(x))
                x.bitwise_or_(y)
                mlu_ptr = x_mlu.data_ptr()
                x_mlu.bitwise_or_(self.to_device(y))
                self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.003,
                                        use_MSE = True)
                self.assertEqual(mlu_ptr, x_mlu.data_ptr())

                # test scalar input
                # FIXME(liuyuxin): scalar computation are not supported for some types
                # will be fixed in future.
                if dtype1 == torch.uint8 or dtype1 == torch.int16 or dtype1 == torch.int8:
                   continue
                input1 = self._generate_tensor(shape1, dtype1)
                input3 = self._generate_scalar(dtype2)
                input1_copy = copy.deepcopy(input1)
                input1.bitwise_or_(input3)
                input1_mlu = self.to_device(input1_copy)
                raw_ptr = input1_mlu.data_ptr()
                input1_mlu.bitwise_or_(input3)
                cur_ptr = input1_mlu.data_ptr()
                self.assertEqual(raw_ptr, cur_ptr)
                self.assertTensorsEqual(input1, input1_mlu.cpu(), 0)

                if input1.dim() == 4:
                    # test channels last
                    x = self._generate_tensor(shape1, dtype1).to(memory_format = torch.channels_last)
                    y = self._generate_tensor(shape2, dtype2)
                    x_mlu = self.to_device(copy.deepcopy(x))
                    x.bitwise_or_(y)
                    mlu_ptr = x_mlu.data_ptr()
                    x_mlu.bitwise_or_(self.to_device(y))
                    self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.003,
                                            use_MSE = True)
                    self.assertEqual(mlu_ptr, x_mlu.data_ptr())
                    self.assertTrue(x.stride() == x_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_or_out(self):
        self_tensor = torch.randint(-16776321, 16776565, (1,2,3,4))
        other = 6654734
        device="mlu"
        cpu_result = torch.ones(5, 2, 3, 4, dtype = torch.int32)
        device_result = torch.ones(5, 2, 3, 4, dtype = torch.int32).to(device)
        device_result1 = torch.ones(5, 2, 3, 4, dtype = torch.int32).to(device, memory_format = torch.channels_last)
        torch.bitwise_or(self_tensor, other, out = cpu_result)
        torch.bitwise_or(self_tensor.to(device), other, out = device_result1)
        torch.bitwise_or(self_tensor.to(device, memory_format = torch.channels_last), other, out = device_result)
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0)
        self.assertTensorsEqual(cpu_result, device_result1.cpu(), 0)

        self_tensor = torch.randint(-16776321, 16776565, (1,2,3,4))
        other = torch.randint(-16776321, 16776565, (4,2,3,4))
        cpu_result = torch.ones(5, 2, 3, 4, dtype = torch.int32)
        device_result = torch.ones(5, 2, 3, 4, dtype = torch.int32).to(device)
        torch.bitwise_or(self_tensor, other, out = cpu_result)
        torch.bitwise_or(self_tensor.to(device, memory_format = torch.channels_last), other.to(device, memory_format = torch.channels_last), out = device_result)
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_or_exception(self):
        ref_msg = r"self and other only support int related types"
        input1 = torch.arange(24, dtype=torch.float).reshape(1, 2, 3, 4)
        input2 = torch.arange(96).reshape(4, 2, 3, -1).float()
        input1_mlu = self.to_device(input1)
        input2_mlu = self.to_device(input2)
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            input1_mlu | input2_mlu

if __name__ == "__main__":
    unittest.main()
