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
    def test_queue(self):
        input_data = torch.randn(1,3,2,2)
        output = torch.abs(input_data.to(ct.mlu_device()))
        queue = ct.current_queue()
        default_queue = ct.default_queue()
        queue.synchronize()
        self.assertTensorsEqual(torch.abs(input_data), output.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_current_queue(self):
        queue0 = ct.current_queue()
        self.assertEqual(queue0.device, 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_device_with_queue(self):
        queue = ct.current_queue()
        self.assertEqual(queue.device, 0)
        self.assertEqual(queue.device_index, 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_default_queue(self):
        if ct.device_count() < 2:
            return
        queue0 = ct.default_queue()
        queue1 = ct.default_queue(1)
        self.assertEqual(queue0.device, 0)
        self.assertEqual(queue1.device, 1)

    #@unittest.skip("not test")
    @testinfo()
    def test_queues(self):
        current_queue = ct.current_queue()
        user_queue = ct.Queue()
        self.assertEqual(ct.current_queue(), current_queue)

    #@unittest.skip("not test")
    @testinfo()
    def test_queue_guard(self):
        current_queue = ct.current_queue()
        for i in range(ct.device_count()):
            with ct.Queue(i):
                assert current_queue != ct.current_queue()
                assert ct.default_queue() != ct.current_queue()

    #@unittest.skip("not test")
    @testinfo()
    def test_device_queue(self):
        user_queue = ct.Queue()
        assert user_queue != ct.current_queue()
        user_queue.synchronize()
        ct.set_device(0)
        notifier = Notifier.Notifier()
        notifier.place(user_queue)
        notifier.synchronize()
        self.assertTrue(notifier.query())


if __name__ == '__main__':
    unittest.main()

