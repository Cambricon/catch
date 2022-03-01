from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu.core.mlu_model as ct   # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_round(self):
        shape_list = [(2,3,4,3,4,2,1),(2,3,4),(1,32,5,12,8),
                      (2,128,10,6),(2,512,8),(1,100),(24,)]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.round(x)
            out_mlu = torch.round(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_channel_last(self):
        shape = (2,128,10,6)
        x = torch.randn(shape, dtype=torch.float).to(memory_format=torch.channels_last)
        out_cpu = torch.round(x)
        out_mlu = torch.round(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_not_dense(self):
        shape_list = [(2,3,4),(1,32,5,12,8),
                      (2,128,10,6)]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.round(x[:, ... , :2])
            out_mlu = torch.round(self.to_mlu(x)[:, ... , :2])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_inplace(self):
        shape_list = [(2,3,4,3,4,2,1),(2,3,4),(1,32,5,12,8),
                      (2,128,10,6),(2,512,8),(1,100),(24,)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to('mlu')
            out_cpu = torch.round_(x_cpu)
            out_mlu = torch.round_(x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to('mlu')
            x_cpu.round_()
            x_mlu.round_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_inplace_channel_last(self):
        shape_list = [(32,5,12,8),
                      (2,128,10,6)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i]).to(memory_format=torch.channels_last)
            x_mlu = x_cpu.to('mlu')
            out_cpu = torch.round_(x_cpu)
            out_mlu = torch.round_(x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

            x_cpu = torch.randn(shape_list[i]).to(memory_format=torch.channels_last)
            x_mlu = x_cpu.to('mlu')
            x_cpu.round_()
            x_mlu.round_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_inplace_not_dense(self):
        shape_list = [(2,3,4),(1,32,5,12,8),
                      (2,128,10,6)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to('mlu')
            out_cpu = torch.round_(x_cpu[:, ... , :2])
            out_mlu = torch.round_(x_mlu[:, ... , :2])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to('mlu')
            x_cpu[:, ... , :2].round_()
            x_mlu[:, ... , :2].round_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    #@unittest.skip("not test")
    @testinfo()
    def test_round_out(self):
        shape_list = [(2,3,4,3,4,2,1),(2,3,4),(1,32,5,12,8),
                      (2,128,10,6),(2,512,8),(1,100),(24,)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to('mlu')
            out_tmpcpu = torch.zeros(shape_list[i])
            out_tmpmlu = torch.zeros(shape_list[i]).to('mlu')
            out_tmpcpu_2 = torch.zeros((1))
            out_tmpmlu_2 = torch.zeros((1)).to('mlu')
            out_cpu = torch.round(x_cpu, out=out_tmpcpu)
            out_mlu = torch.round(x_mlu, out=out_tmpmlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(out_tmpcpu, out_tmpmlu.cpu(), 0)
            out_cpu_2 = torch.round(x_cpu, out=out_tmpcpu_2)
            out_mlu_2 = torch.round(x_mlu, out=out_tmpmlu_2)
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(), 0)
            self.assertTensorsEqual(out_tmpcpu_2, out_tmpmlu_2.cpu(), 0)

if __name__ == "__main__":
    unittest.main()
