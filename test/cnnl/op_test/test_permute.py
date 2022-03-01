from __future__ import print_function

import sys
import os
# import copy
# import time
import unittest
import logging
# import numpy as np

import torch
import torch_mlu.core.mlu_model as ct       # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_permute(self):
        shape_permute = [[(3, 224, 224), (0, 2, 1)],
                         [(2, 3, 224, 224), (0, 3, 1, 2)],
                         [(2, 10, 3, 224, 224), (0, 4, 1, 2, 3)],
                         [(2, 3, 10, 3, 224, 224), (0, 4, 5, 1, 2, 3)]]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in data_types:
            for shape, permute_index in shape_permute:
                input_t = torch.rand(shape)
                input_mlu = self.to_mlu_dtype(input_t, data_type)
                output_cpu = input_t.permute(permute_index)
                output_mlu = input_mlu.permute(permute_index)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_permute_channel_last(self):
        shape_permute = [[(3, 224, 224), (0, 2, 1)],
                         [(2, 3, 224, 224), (0, 3, 1, 2)],
                         [(2, 10, 3, 224, 224), (0, 4, 1, 2, 3)],
                         [(2, 3, 10, 3, 224, 224), (0, 4, 5, 1, 2, 3)]]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in data_types:
            for shape, permute_index in shape_permute:
                if len(shape) == 4:
                    memory_type = torch.channels_last
                elif len(shape) == 5:
                    memory_type = torch.channels_last_3d
                else:
                    memory_type = torch.contiguous_format
                input_t = torch.rand(shape).to(memory_format=memory_type)
                input_mlu = self.to_mlu_dtype(input_t, data_type)
                output_cpu = input_t.permute(permute_index)
                output_mlu = input_mlu.permute(permute_index)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_permute_not_dense_and_channel_last(self):
        shape_permute = [[(3, 224, 224), (0, 2, 1)],
                         [(2, 3, 224, 224), (0, 3, 1, 2)],
                         [(2, 10, 3, 224, 224), (0, 4, 1, 2, 3)],
                         [(2, 3, 10, 3, 224, 224), (0, 4, 5, 1, 2, 3)]]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in data_types:
            for shape, permute_index in shape_permute:
                if len(shape) == 4:
                    memory_type = torch.channels_last
                elif len(shape) == 5:
                    memory_type = torch.channels_last_3d
                else:
                    memory_type = torch.contiguous_format
                input_t = torch.rand(shape).to(memory_format=memory_type)
                input_mlu = self.to_mlu_dtype(input_t, data_type)
                output_cpu = input_t[..., :112].permute(permute_index)
                output_mlu = input_mlu[..., :112].permute(permute_index)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err, use_MSE=True)

if __name__ == "__main__":
    unittest.main()
