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

class TestNotifier(TestCase):
    def test_place(self):
        input1 = torch.randn(1,3,2,2).to(ct.mlu_device())
        output = torch.neg(input1)
        notifier = Notifier.Notifier()
        notifier.place()

    def test_query(self):
        input1 = torch.randn(1,3,2,2).to(ct.mlu_device())
        input2 = torch.randn(1,3,2,2).to(ct.mlu_device())
        output = torch.neg(input1)
        notifier = Notifier.Notifier()
        input3 = input1 + input2
        self.assertTrue(notifier.query())

    def test_synchronize_elapsed_time(self):
        input1 = torch.randn(1000,1000,2,2).to('mlu')
        input2 = torch.randn(1000,1000,2,2).to('mlu')
        output = torch.neg(input1)
        start = Notifier.Notifier()
        end= Notifier.Notifier()
        start.place()
        for i in range(10):
            input3 = input1 * input2
            input1 = input3 * input2
        end.place()
        end.synchronize()
        e2e_time_ms = start.elapsed_time(end)
        hardware_time_ms = start.hardware_time(end) / 1000.0
        diff_ms = e2e_time_ms - hardware_time_ms
        self.assertTrue(diff_ms >= 0)

    def test_synchronize_hardware_time(self):
        input1 = torch.randn(1000,3,2,2).to(ct.mlu_device())
        input2 = torch.randn(1000,3,2,2).to(ct.mlu_device())
        output = torch.neg(input1)
        start = Notifier.Notifier()
        end= Notifier.Notifier()
        start.place()
        input3 = input1 * input2
        start.place()
        input1 = input3* input2
        end.place()
        end.synchronize()
        time = start.hardware_time(end)
        self.assertTrue(time > 0)

    def test_wait(self):
        start = Notifier.Notifier()
        queue = ct.current_queue()
        user_queue = ct.Queue()
        start.place(queue)
        start.wait(user_queue)
        user_queue.synchronize()
        self.assertTrue(start.query())
        self.assertTrue(queue.query())
        self.assertTrue(user_queue.query())

if __name__ == '__main__':
    unittest.main()
