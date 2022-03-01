from __future__ import print_function

import sys
import os
import unittest
import logging
import numpy as np

import torch
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

class TestAsTensor(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_as_tensor(self):
        x = [[0, 1], [2, 3]]
        y = torch.tensor(x)
        # from tensor (doesn't copy unless type is different)
        self.assertIsNot(y, torch.as_tensor(y, device='mlu'))
        y_mlu = y.to('mlu')
        self.assertIs(y_mlu, torch.as_tensor(y_mlu))
        self.assertIs(y_mlu, torch.as_tensor(y_mlu, device='mlu'))

        n = np.random.randn(5, 6)
        n_astensor = torch.as_tensor(n, device='mlu')
        self.assertEqual(torch.tensor(n, device='mlu').cpu(), n_astensor.cpu())


if __name__ == '__main__':
    unittest.main()
