from __future__ import print_function

import sys
import os
# import copy
# import time
import unittest
import logging

import torch
# import torch.nn as nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
from common_utils import testinfo, TestCase  # pylint: disable=C0413
import torch_mlu.core.mlu_model as ct # pylint: disable=C0413, W0611
logging.basicConfig(level=logging.DEBUG)


class TestEmpty(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_empty(self):
        shape_memory_format_list = [((2, 3, 4, 5), torch.channels_last),
                                    ((6, 7, 8, 9), torch.contiguous_format),
                                    ((5, 4, 3, 2), torch.contiguous_format),
                                    ((64, 3, 224, 224), torch.channels_last),
                                    ((5, 0, 3, 2), torch.contiguous_format),
                                    ((2, 3, 0, 5), torch.channels_last),
                                   ]
        for shape, memory_format in shape_memory_format_list:
            x_mlu = torch.empty(shape, memory_format=memory_format, device='mlu')
            x_mlu_to_cpu = torch.empty(shape, memory_format=memory_format, device='mlu').cpu()
            x_cpu = torch.empty(shape, memory_format=memory_format)
            self.assertEqual(x_cpu.size(), x_mlu.size())
            self.assertEqual(x_cpu.size(), x_mlu_to_cpu.size())
            self.assertEqual(x_cpu.stride(), x_mlu.stride())
            self.assertEqual(x_cpu.stride(), x_mlu_to_cpu.stride())


if __name__ == '__main__':
    unittest.main()
