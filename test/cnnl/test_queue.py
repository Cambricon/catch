from __future__ import print_function
import torch
import torch.nn as nn
import torch_mlu
import torch_mlu.core.mlu_model as ct
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import sys
import os
import copy

import time
import unittest
sys.path.append("../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)

import torch_mlu.core.device.notifier as Notifier

class TestQueue(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_wait_queue(self):
        x = torch.randn(10000000, dtype=torch.float)
        notifier = Notifier.Notifier()
        with ct.Queue(0):
            x_mlu = x.to('mlu:0', non_blocking=True)
            notifier.place(ct.current_queue())
        with ct.Queue(0):
            notifier.wait(ct.current_queue())
            x_add = x_mlu + 1
        self.assertTensorsEqual(x + 1, x_add.cpu(), 0.0, use_MSE=True)


if __name__ == '__main__':
    unittest.main()

