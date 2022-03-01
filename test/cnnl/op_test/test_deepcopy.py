from __future__ import print_function

import sys
import logging
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
import unittest
from copy import deepcopy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

# NOTE: tensor.storage() is not supported for MLU and
# ops such as view does not share MLU storage. So MLU
# deepcopy does not preserve shared memory.
class TestDeepcopyOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_deepcopy_parameter(self):
        l = torch.nn.Linear(10, 1).to('mlu')
        s = l.state_dict(keep_vars=True)
        self.assertEqual(torch.nn.Parameter, type(s['weight']))
        self.assertEqual(torch.nn.Parameter, type(s['bias']))

        s2 = deepcopy(s)
        self.assertEqual(torch.nn.Parameter, type(s2['weight']))
        self.assertEqual(torch.nn.Parameter, type(s2['bias']))
        self.assertEqual(s['weight'], s2['weight'])
        self.assertEqual(s['bias'], s2['bias'])

    # @unittest.skip("not test")
    @testinfo()
    def test_deepcopy_channels_last(self):
        x = torch.rand(2, 3, 4, 5).to(memory_format=torch.channels_last)
        x_mlu = self.to_mlu(x)
        out_cpu = deepcopy(x)
        out_mlu = deepcopy(x_mlu)
        self.assertEqual(out_cpu, out_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_deepcopy_no_dense(self):
        x = torch.rand(2, 3, 4, 5)
        x_mlu = self.to_mlu(x)[..., ::2]
        x = x[..., ::2]
        out_cpu = deepcopy(x)
        out_mlu = deepcopy(x_mlu)
        self.assertEqual(out_cpu, out_mlu)


if __name__ == '__main__':
    unittest.main()
