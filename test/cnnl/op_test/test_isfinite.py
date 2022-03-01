from __future__ import print_function

import sys
import logging
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
import numpy
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestIsfiniteOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_isinf_isnan(self, device='mlu', dtype=torch.float):
        vals = (-float('inf'), float('inf'), float('nan'), -1, 0, 1)
        self.compare_with_numpy(torch.isfinite, numpy.isfinite, vals, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_isinf_isnan_with_cpu(self, dtype=torch.float):
        vals = (-float('inf'), float('inf'), float('nan'), -1, 0, 1)
        x = torch.tensor(vals, dtype=dtype)
        res_cpu = torch.isfinite(x)
        res_mlu = torch.isfinite(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_empty_tensor(self, dtype=torch.float):
        vals = torch.rand([0, 2, 3])
        x = torch.tensor(vals, dtype=dtype)
        res_cpu = torch.isfinite(x)
        res_mlu = torch.isfinite(self.to_mlu(x))
        self.assertTensorsEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_isinf_isnan_int(self, device='mlu', dtype=torch.long):
        vals = (-1, 0, 1)
        self.compare_with_numpy(torch.isfinite, numpy.isfinite, vals, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_channels_last(self, dtype=torch.float):
        x = torch.randn((3,4,5,6), dtype=dtype).to(memory_format=torch.channels_last)
        x[0][1][2][3] = torch.tensor(-float('inf'), dtype=dtype)
        x[1][2][3][4] = torch.tensor(float('inf'), dtype=dtype)
        x[2][3][4][5] = torch.tensor(float('nan'), dtype=dtype)
        res_cpu = torch.isfinite(x)
        res_mlu = torch.isfinite(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_not_dense(self, dtype=torch.float):
        x = torch.randn((3,4,5,6), dtype=dtype).to(memory_format=torch.channels_last)
        x[0][1][2][3] = torch.tensor(-float('inf'), dtype=dtype)
        x[1][2][3][4] = torch.tensor(float('inf'), dtype=dtype)
        x[2][3][4][5] = torch.tensor(float('nan'), dtype=dtype)
        res_cpu = torch.isfinite(x[..., :4])
        res_mlu = torch.isfinite(self.to_mlu(x)[..., :4])
        self.assertEqual(res_cpu, res_mlu.cpu())
if __name__ == '__main__':
    unittest.main()
