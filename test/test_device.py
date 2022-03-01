from __future__ import print_function

import sys
import logging
import unittest
import numpy as np
import torch
import torch_mlu.core.mlu_model as ct
sys.path.append("../")
from common_utils import TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestDevice(TestCase):
    def test_device(self):
        device_count = ct.device_count()
        for device_id in np.arange(0, device_count, 1):
            ct.set_device(device_id)
        input = torch.randn(8,3,24,24).to(ct.mlu_device())
        input1 = torch.abs(input)
        self.assertEqual(input1.device.index, ct.get_device())

    def test_device_by_tensor(self):
        ct.set_device(0)
        input = torch.randn(8,3,24,24).to(ct.mlu_device())
        input1 = torch.abs(input)
        input2 = torch.randn(4,3,24,24).to(input1.device)
        self.assertEqual(input1.device.index, input2.device.index)

        device_count = ct.device_count()
        if device_count > 1:
            input_new = torch.randn(8,3,22,22).to("mlu:0")
            input1_new = torch.abs(input_new)
            input2_new = torch.randn(4,3,22,22).to("mlu:1")
            self.assertNotEqual(input1_new.device.index, input2_new.device.index)

    def test_device_count(self):
        device_count = ct.device_count()
        self.assertLessEqual(0, device_count, '')

    def test_with_device(self):
        ct.set_device(0)
        self.assertEqual(0, ct.get_device())
        for i in range(ct.device_count()):
            with ct.Device(i):
                self.assertEqual(i, ct.get_device())
        self.assertEqual(0, ct.get_device())

    def test_device_synchronize(self):
        ct.synchronize()
        ct.synchronize('mlu')
        ct.synchronize('mlu:0')
        for i in range(ct.device_count()):
            ct.synchronize(i)

        with self.assertRaisesRegex(ValueError, "Expected a cuda or mlu device, but"):
            ct.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected a cuda or mlu device, but"):
            ct.synchronize("cpu")


if __name__ == '__main__':
    unittest.main()
