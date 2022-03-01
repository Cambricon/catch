from __future__ import print_function
import sys
from sys import path
import os
from os.path import dirname
import torch
import torch.nn as nn
from torch.utils.data import _utils, Dataset, IterableDataset, TensorDataset, DataLoader, ConcatDataset, ChainDataset
import torch_mlu
import torch_mlu.core.mlu_model as ct



path.append(dirname(path[0]))
import unittest
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
import time

class TestPinMemory(TestCase):

    def setUp(self):
        super(TestPinMemory, self).setUp()
        self.data = torch.randn(100, 3, 10, 10)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)

    @staticmethod
    def _test_generator(self, num_workers, pin_memory, non_blocking, use_io_queue):
        if use_io_queue:
            os.environ['USE_IO_QUEUE'] = 'ON'
        else:
            if os.environ['USE_IO_QUEUE'] is not None:
                del os.environ['USE_IO_QUEUE']
        loader = DataLoader(self.dataset, batch_size=2, num_workers=num_workers, pin_memory=pin_memory)
        for input, target in loader:
            if not use_io_queue:
                self.assertTrue(pin_memory == input.is_pinned())
                self.assertTrue(pin_memory == target.is_pinned())

            input_mlu = input.to(ct.mlu_device(), non_blocking=non_blocking)
            target_mlu = target.to(ct.mlu_device(), non_blocking=non_blocking)
            input_mlu *=1
            target_mlu *=1
            input_cpu = input_mlu.cpu()
            target_cpu = target_mlu.cpu()
            self.assertTensorsEqual(input.cpu(),input_cpu,0.,use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_dataloader(self):
        for pin_memory in [True, False]:
            for num_workers in [0,1]: #0 is single process; 1 is multi process
                for non_blocking in [True, False]:
                    for use_io_queue in [True, False]:
                        self._test_generator(self, num_workers, pin_memory, non_blocking, use_io_queue)

if __name__ == '__main__':
    unittest.main()
