from __future__ import print_function

import sys
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=all
import unittest
import logging
import copy

import numpy as np
import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)

class TestIndexPut(TestCase):
    def _generate_tensor(self, shape, dtype, value = 16777216, reduce_dim = False):
        if dtype == torch.bool:
            out = torch.randint(2, shape).type(dtype)
        else:
            # TODO(liuyuxin): test negative number in future.
            out = torch.randint(value, shape).type(dtype)
        if len(shape) == 1 and reduce_dim:
            out = out[0]
        return out

    #@unittest.skip("not test")
    @testinfo()
    def test_index_put_network(self):
        param_list = [((4, 30000), [(4,),(4,)], torch.long, torch.randn(4)),
                      ((1, 3, 2560), [(1, 3)], torch.bool, 1.1),
                      ((2048, ), [(2048,)], torch.bool, -1),
                      ((8732, ), [(1,)], torch.long, -1),
                      ((8732, ), [(0,)], torch.long, -1),
                      ((0, ), [(0,)], torch.long, -1),
                      ((55, 555), [(55, 555)], torch.bool, 1.1),
                      ((8, 8732), [(8, 8732)], torch.bool, 1.1),
                      ((16, 16, 16), [(4, 3)], torch.long, 100),
                      ((512, 256, 7, 7), [(0,)], torch.long, torch.randn([0, 256, 7, 7])),
                      ((1, 1, 2560), [(1, 1)], torch.bool, 1.1)]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = self._generate_tensor(input_shape, torch.float)
            input_mlu = self.to_device(input)
            indices = []
            indices_mlu = []
            for i in range(len(indices_shape)):
                if input_shape[i] == 0:
                    indice = self._generate_tensor(indices_shape[i], type_)
                else:
                    indice = self._generate_tensor(indices_shape[i], type_, input_shape[i])
                indices.append(indice)
                indices_mlu.append(self.to_device(indice))
            input[indices] = value_
            input_mlu[indices_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        param_list = [((4, 2), (1,), torch.long , torch.randn(2))]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[0])
            indice_mlu = self.to_device(indice)
            input[indice] = value_
            input_mlu[indice_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        # with undefined indices
        param_list = [((2, 2), (1,), torch.long , torch.randn(2, 1))]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[1])
            indice_mlu = self.to_device(indice)
            input[:, indice] = value_
            input_mlu[:, indice_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        param_list = [((2, 2), (1,), torch.long , 1)]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[0])
            indice_mlu = self.to_device(indice)
            input[indice, :] = value_
            input_mlu[indice_mlu, :] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        param_list = [((32, 3, 40, 84, 85), (2,), torch.long , 1)]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[4])
            indice_mlu = self.to_device(indice)
            input[..., indice] = value_
            input_mlu[..., indice_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        # with multiple indices
        # repeatedly writing-in may cause unexpected error,
        # so make sure writing with the same value into same place.
        param_list = [((16, 3, 13, 13), (132,), torch.long , torch.randn(132))]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            repeat_num = indices_shape[0]//input_shape[1]
            indice_1 = torch.randperm(input_shape[1], dtype = type_).repeat(repeat_num)
            indice_2 = torch.randperm(input_shape[1], dtype = type_).repeat(repeat_num)
            indice_3 = torch.randperm(input_shape[1], dtype = type_).repeat(repeat_num)
            indice_4 = torch.randperm(input_shape[1], dtype = type_).repeat(repeat_num)
            value_ = torch.randn(input_shape[1]).repeat(repeat_num)
            input[indice_1, indice_2, indice_3, indice_4] = value_
            input_mlu[self.to_device(indice_1), self.to_device(indice_2),\
                      self.to_device(indice_3), self.to_device(indice_4)] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        # accumulate = True
        param_list = [((8, 8732, 4), [(8, 8732, 4)], torch.bool, 1)]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape).float()
            input_mlu = self.to_device(input)
            indices = []
            indices_mlu = []
            for i in range(len(indices_shape)):
                indice = self._generate_tensor(indices_shape[i], type_, input_shape[i])
                indices.append(indice)
                indices_mlu.append(self.to_device(indice))
            output = torch.index_put(input, indices, torch.tensor(value_).float(), True)
            output_mlu = torch.index_put(input_mlu, indices_mlu,\
                                         self.to_device(torch.tensor(value_).float()), True)
            self.assertTensorsEqual(output, output_mlu.cpu(), 0.00, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_index_put_not_contiguous(self):
        shape_index = [((8, 6), [1, 3, 6]),
                       ((13, 15, 14), [1, 3, 5, 8]),
                       ((53, 71, 3, 3), [2, 4, 6, 7, 8])                       
                      ,]
        for shape, index in shape_index:
            x = torch.randn(shape).float()
            x_mlu = x.to("mlu")
            x_mlu_ptr = x_mlu.data_ptr()
            indices = torch.tensor(index).long()
            indices_mlu = indices.to("mlu")
            value = torch.randn(x[indices, :5].shape).float()
            value_mlu = value.to("mlu")
            x[indices, :5] = value
            x_mlu[indices_mlu, :5] = value_mlu
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.00)
            self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())

    #@unittest.skip("not test")
    @testinfo()
    def test_index_put_exception(self):
        a = self.to_device(torch.rand(2, 3, 4))
        indice1 = self.to_device(torch.tensor([0, 1, 0]))
        indice2 = torch.tensor([0, 1])
        val = torch.randn(1)
        ref_msg_0 = r"^indices have more indices than self dim"
        with self.assertRaisesRegex(RuntimeError, ref_msg_0):
            a.index_put_((indice1, self.to_device(a[0,:,:]>0), indice1, indice1), values=self.to_device(val))
        ref_msg_2 = r"^self and values must have same dtype"
        with self.assertRaisesRegex(RuntimeError, ref_msg_2):
            a.index_put_((indice1, self.to_device(a[0,:,:]>0)), values=self.to_device(val.int()))
        ref_msg_4 = r"^indices can't be empty"
        with self.assertRaisesRegex(RuntimeError, ref_msg_4):
            a.index_put_(indices=self.to_device(None), values=self.to_device(val))
        ref_msg_5 = r"^support only int, bool and long"
        with self.assertRaisesRegex(RuntimeError, ref_msg_5):
            a.index_put_((indice1.float(),), values=self.to_device(val))

if __name__ == '__main__':
    unittest.main()
