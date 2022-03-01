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

class TestCachingAllocator(TestCase):

    @staticmethod
    def _test_memory_generator(self, device=None, N=35):
        # init device
        torch.abs(torch.Tensor(1).to(ct.mlu_device()))
        if device is None:
            device = ct.current_device()

        m0 = ct.memory_allocated(device)
        last_m_arr = [ct.memory_allocated(device)]
        max_m_arr = [ct.max_memory_allocated(device)]
        last_c_arr = [ct.memory_cached(device)]
        max_c_arr = [ct.max_memory_cached(device)]

        def allocate(*size):
            ct.set_device(device)
            t = torch.Tensor(*size).float().to(ct.mlu_device())
            return t+0

        def assert_change(comp=1, empty_cache=False):
            # comp > 0: increased
            # comp = 0: equal
            # comp < 0: decreased
            new_m = ct.memory_allocated(device)
            new_max_m = ct.max_memory_allocated(device)
            if comp > 0:
                self.assertGreater(new_m, last_m_arr[0])
            elif comp < 0:
                self.assertLess(new_m, last_m_arr[0])
            else:
                self.assertEqual(new_m, last_m_arr[0])
            self.assertLessEqual(new_m, new_max_m)
            self.assertGreaterEqual(new_max_m, max_m_arr[0])
            last_m_arr[0] = new_m
            max_m_arr[0] = new_max_m

            new_c = ct.memory_cached(device)
            new_max_c = ct.max_memory_cached(device)
            self.assertLessEqual(new_c, new_max_c)
            self.assertGreaterEqual(new_max_c, max_c_arr[0])
            last_c_arr[0] = new_c
            max_c_arr[0] = new_max_c

            if empty_cache:
                ct.empty_cached_memory()
                new_c = ct.memory_cached(device)
                new_max_c = ct.max_memory_cached(device)
                self.assertLessEqual(new_c, last_c_arr[0])
                self.assertLessEqual(new_c, new_max_c)
                self.assertEqual(new_max_c, max_c_arr[0])
                last_c_arr[0] = new_c

        assert_change(0)
        assert_change(0, empty_cache=True)
        assert_change(0)
        yield

        tensors1 = [allocate(1), allocate(10, 20), allocate(200, 300, 2000)]

        #check the memory status of empty device
        device_count = ct.device_count()
        for i in range(device_count):
            if i==device:
                self.assertEqual(ct.memory_cached(i), ct.memory_cached())
                self.assertEqual(ct.max_memory_cached(i), ct.max_memory_cached())
                self.assertEqual(ct.memory_allocated(i), ct.memory_allocated())
                self.assertEqual(ct.max_memory_allocated(i), ct.max_memory_allocated())
                continue
            self.assertEqual(ct.memory_cached(i), 0)
            self.assertEqual(ct.max_memory_cached(i), 0)
            self.assertEqual(ct.memory_allocated(i), 0)
            self.assertEqual(ct.max_memory_allocated(i), 0)

        m1 = ct.memory_allocated(device)
        assert_change(1)
        yield

        tensors2 = []

        # small chunks with allocation smaller than 1MB
        for i in range(1, int(N / 2) + 1):
            # small ones
            tensors2.append(allocate(i, i * 4))
            assert_change(1)
            yield

        # large chunks with allocation larger than 1MB
        for i in range(5, int(N / 2) + 5):
            tensors2.append(allocate(i, i * 7, i * 9, i * 11))
            assert_change(1)
            yield

        # TODO: unsupported empty tensor calculation
        # tensors2.append(allocate(0, 0, 0))
        # assert_change(0)
        # yield

        permute = []
        for i in torch.randperm(len(tensors2)):
            permute.append(tensors2[i])
            assert_change(0)
            yield

        del tensors2
        # now the memory of tensor2 is used by permute
        assert_change(0)
        yield
        tensors2 = permute
        assert_change(0)
        yield
        del permute
        # now the memory of permute is used by tensor2
        assert_change(0)
        yield



        for i in range(int(N / 2)):
            x = tensors2[i].numel()
            del tensors2[i]
            assert_change(-x)  # in case that tensors2[i] is empty
            yield

        for i in range(2, int(2 * N / 3) + 2):
            tensors2.append(allocate(i, i * 3, i * 8))
            assert_change(1)
            yield

        del tensors2
        assert_change(-1)
        assert_change(0)
        self.assertEqual(ct.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1)
        self.assertEqual(ct.memory_allocated(device), m0)

        if int(os.environ.get("ENABLE_CATCH_MEMORY_DEBUG")) :
            t3 = allocate(100)
            ct.memory_debug(t3)
            ct.memory_debug()
            assert_change(1)
        # test empty_cache
        assert_change(0, empty_cache=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_memory_stats(self):
        os.environ["ENABLE_CATCH_MEMORY_DEBUG"] = '0'
        ct.empty_cached_memory()
        for _ in self._test_memory_generator(self):
            pass
        os.environ["ENABLE_CATCH_MEMORY_DEBUG"] = '1'
        for _ in self._test_memory_generator(self):
            pass
        os.environ["ENABLE_CATCH_MEMORY_DEBUG"] = '0'

if __name__ == '__main__':
    unittest.main()
