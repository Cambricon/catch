from __future__ import print_function
import logging
import os
import sys
import unittest
os.environ["DEFAULT_MLU_DEVICE_NAME"] = "MLU270"  # pylint: disable=C0413

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestSetDevice(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_device_name(self):
        assert not ct.is_using_floating_device()

if __name__ == '__main__':
    unittest.main()
