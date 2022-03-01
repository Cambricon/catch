from __future__ import print_function
import os
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=all

import sys
import logging
import unittest
import torch
import torch_mlu.core.mlu_model as ct       # pylint: disable=W0611
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

class TestExpandOp(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_expand(self):
        for shape1,shape2 in [
                            ((5),(256,5)),
                            ((4,5),(1,3,4,5)),
                            ((4,5),(0,4,5)),
                            ((2,3,4),(3,-1,3,4)),
                            ((128,1,1024),(-1,379,-1)),
                            ((2048,5),(1,3,2048,5)),
                            ((24,1,1,1),(24,51,51,1)),
                            ((2,6,1,1),(24,51,2,6,1,1)),
                            ((7,56),(256,0,7,56)),
                            ((2048,5),(0,3,2048,5)),
                            ((8,1,64,64),(8,512,64,64)),
                              ]:
            x = torch.randn(shape1, dtype=torch.float)
            out_cpu = x.expand(shape2) * 3.14
            out_mlu = self.to_mlu(x).expand(shape2) * 3.14
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_expand_channel_last(self):
        for shape1,shape2 in [((24,1,1,1),(24,51,51,1)),
                            ((2,6,1,1),(24,51,2,6,1,1))]:
            x = torch.randn(shape1, dtype=torch.float).to(memory_format=torch.channels_last)
            out_cpu = x.expand(shape2)
            out_mlu = self.to_mlu(x).expand(shape2)
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_expand_not_dense(self):
        for shape1,shape2 in [((4,5),(1,3,4,3)),
                            ((4,5),(0,4,3))]:
            x = torch.randn(shape1, dtype=torch.float)
            out_cpu = x[:,:3].expand(shape2)
            out_mlu = self.to_mlu(x)[:,:3].expand(shape2)
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE = True)

    #@unittest.skip("not test")
    @testinfo()
    def test_expand_exception(self):
        a = torch.randn((2,2,3,3), dtype=torch.float).to('mlu')
        ref_msg = r"^expand\(MLUFloatType\{\[2, 2, 3, 3\]\}, size=\[2, 3, 3\]\): the number"
        ref_msg = ref_msg + r" of sizes provided \(3\) must be greater or equal to the number"
        ref_msg = ref_msg + r" of dimensions in the tensor \(4\)$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.expand((2,3,3))

if __name__ == '__main__':
    unittest.main()
