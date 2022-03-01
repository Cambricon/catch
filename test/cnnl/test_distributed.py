from __future__ import absolute_import, division, print_function, unicode_literals
#pylint: disable=C0413,C0411
import copy
import multiprocessing
import os
import sys
from sys import path
from os.path import dirname
from itertools import product
import time
import unittest
from functools import reduce, wraps
import distutils.dir_util
from argparse import ArgumentParser
from multiprocessing.managers import BaseManager
from queue import Queue
import random as rd

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group


import torch_mlu.core.mlu_model as ct

path.append(dirname(path[0]))
from common_utils import TestCase

INIT_METHOD = os.getenv("INIT_METHOD", "env://")
DEFAULT_TIMEOUT = 300
CUSTOMIZED_TIMEOUT = {"test_distributedDataParallel":200, "test_pressure":200, "test_barrier":300}
SKIP_IF_BACKEND_UNAVAILABLE = 78
cwd = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(cwd, "tmp")


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Catch distributed training unittest")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes participate in testing, "
                             "this is set ot 1 by default.")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node testing, "
                             "this is set to 0 by default.")
    parser.add_argument("--nproc_per_node", type=int, default=2,
                        help="The number of processes to launch on each node, "
                             "for multi-node testing, this is set to 2 by default.")
    parser.add_argument("--connects", default=rd.randint(-1, 4), type=int, choices=range(-1, 4),
                        help="We support testing for different technologies of "
                             "connection. Different techs have different priority "
                             "levels. In this script, C2C > P2P > SHM > SOCKET. "
                             "when input is -1, no cncl environment will be set; "
                             "input is 0, all techs can be used; input is 1, only "
                             "P2P, SHM and SOCKET can be used, C2C is prohibited; "
                             "2: SHM, SOCKET; 3: SOCKET. By default, every techs "
                             "have chances to be tested. Note: When here are multi "
                             "node, connects would be set to -1 forcibly, please "
                             "make sure different node use the same cncl environments "
                             "according to cncl user guide doc.")
    parser.add_argument("--master_addr", default="127.0.0.5", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1, now this is "
                             "set to 127.0.0.5 by default.")
    parser.add_argument("--master_port", default=29400, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training, now this is set to 29400 by default. "
                             "In addition, we also use (master_port + 10) for sync.")
    parser.add_argument("--delay_time", default=0, type=int,
                        help="The communication time may be different between "
                             "different environment. So we provide a choice for "
                             "user to add additional delay time for all test case.")

    parser.add_argument('unittest_args', nargs='*')

    return parser.parse_args()

def _build_tensor(size, value=None):
    if value is None:
        value = size
    return torch.FloatTensor(size, size, size).fill_(value)

def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT

class QueueManager(BaseManager):
    pass

class Linear_mlu(nn.Linear):   # pylint: disable=W0223
    def forward(self, input_, flg):
        if flg:
            if self.bias is not None:
                bias_cpu = self.bias.cpu()
            else:
                bias_cpu = None
            return F.linear(input_, self.weight.cpu(), bias_cpu)
        else:
            return F.linear(input_, self.weight, self.bias)

class Conv2d_mlu(nn.Conv2d):  # pylint: disable=W0223
    def forward(self, input_, flg):
        if flg:
            return self._conv_forward(input_, self.weight.cpu())
        else:
            return self._conv_forward(input_, self.weight)

class _FC2(nn.Module):   # pylint: disable=W0223
    def __init__(self):
        super(_FC2, self).__init__()   # pylint: disable=R1725
        self.fc = Linear_mlu(10, 12, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x, flg):
        x = self.fc(x, flg)
        return x

class Net(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(Net, self).__init__()   # pylint: disable=R1725
        self.fc1 = Linear_mlu(2, 10, bias=False)
        self.fc2 = _FC2()
        self.conv = Conv2d_mlu(2, 6, 2, bias=False)
        self.fc3 = Linear_mlu(12, 4, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.Tensor([2, 2]).long(), requires_grad=False)

    def forward(self, x, flg):
        x = self.relu(self.fc1(x, flg))
        x = self.relu(self.fc2(x, flg))
        x = self.conv(x.view(-1, 2, 3, 2), flg).view(-1, 12)
        x = self.fc3(x, flg)
        return F.softmax(x, dim=1)

class _DistTestBase(object):  # pylint: disable=R0205
    args = None
    rank = 0
    world_size = 0

    @classmethod
    def _init_global_test(cls):
        group = [i for i in range(0, dist.get_world_size())]  # pylint: disable=R1721
        rank = dist.get_rank()
        return (group, rank)

    def _test_broadcast_helper(self, group, rank):
        ct.set_device(rank % ct.device_count())
        for ttype, value, is_test in [
            ("torch.FloatTensor", -1e-10, True),
            ("torch.HalfTensor", -0.1, True),
            ("torch.CharTensor", -2, True),
            ("torch.ByteTensor", 129, True),
            ("torch.IntTensor", -1e5, True),
            ("torch.LongTensor", 1e5, True),
            ("torch.DoubleTensor", -1e-10, True),
        ]:
            if not is_test:
                continue
            for src in group:
                #expected_tensor = self.to_device(torch.tensor([]))
                expected_tensor = self.to_device(_build_tensor(src + 1, value).type(ttype))
                if rank == src:
                    dist.broadcast(expected_tensor, src)
                else:
                    tensor = self.to_device(_build_tensor(src + 1, -1).type(ttype))
                    dist.broadcast(tensor, src)
                    self.assertEqual(tensor.size(), expected_tensor.size())
                    self.assertTrue(tensor.type(torch.float).eq(
                        expected_tensor.type(torch.float)).cpu().min().item())

    #@unittest.skip("not test")
    def test_broadcast(self):
        group, rank = self._init_global_test()
        self._test_broadcast_helper(group, rank)

    def _test_async_helper(self, group, rank, op, master_value,
                           worker_value, expected_value):
        ct.set_device(rank % ct.device_count())
        for src in group:
            if rank == src:
                tensor = self.to_device(_build_tensor(src + 1, master_value))
                work = dist.all_reduce(tensor, op, async_op=True)
                work.wait()
                self.assertTensorsEqual(
                    tensor.cpu(), _build_tensor(src + 1, expected_value), 3e-3)
            else:
                tensor = self.to_device(_build_tensor(src + 1, worker_value))
                work = dist.all_reduce(tensor, op, async_op=True)
                work.wait()
                self.assertTensorsEqual(
                    tensor.cpu(), _build_tensor(src + 1, expected_value), 3e-3)
            self.assertTrue(work.is_completed())
            self.assertTrue(work.is_success())

    #@unittest.skip("not test")
    def test_async(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = torch.randn(1).item()
        b = torch.randn(1).item()
        self._test_async_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    def _test_reduce_helper(self, group, rank, op, master_value,
                            worker_value, expected_value):
        ct.set_device(rank % ct.device_count())
        for src in group:
            if rank == src:
                tensor = self.to_device(_build_tensor(src + 1, master_value))
                dist.reduce(tensor, src, op)
                self.assertLess((tensor.float() - _build_tensor(src + 1, expected_value).to(
                    ct.mlu_device())).abs().cpu().max().item(), 3e-3)
            else:
                tensor = self.to_device(_build_tensor(src + 1, worker_value))
                dist.reduce(tensor, src, op)

    #@unittest.skip("not test")
    def test_reduce_sum(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = torch.randn(1).item()
        b = torch.randn(1).item()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    #@unittest.skip("not test")
    def test_reduce_product(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.PRODUCT,
            10,
            2,
            reduce((lambda x, y: x * y), [2] * (len(group) - 1), 10),
        )

    #@unittest.skip("not test")
    def test_reduce_min(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MIN,
            1,
            1010,
            1,
        )

    #@unittest.skip("not test")
    def test_reduce_max(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MAX,
            10,
            -1,
            10,
        )

    def _test_all_reduce_helper(self, group, rank, op, master_value,
                                worker_value, expected_value):
        ct.set_device(rank % ct.device_count())
        for src in group:
            if rank == src:
                tensor = self.to_device(_build_tensor(src + 1, master_value))
                dist.all_reduce(tensor, op)
                #print("sum", rank, src, tensor.cpu().view(-1)[0].item())
                self.assertLess((tensor.float() - _build_tensor(src + 1, expected_value).to(
                    ct.mlu_device())).abs().cpu().max().item(), 3e-3)
            else:
                tensor = self.to_device(_build_tensor(src + 1, worker_value))
                dist.all_reduce(tensor, op)
                #print("sum", rank, src, tensor.cpu().view(-1)[0].item())
                self.assertLess((tensor.float() - _build_tensor(src + 1, expected_value).to(
                    ct.mlu_device())).abs().cpu().max().item(), 3e-3)

    #@unittest.skip("not test")
    def test_all_reduce_sum(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = torch.randn(1).item()
        b = torch.randn(1).item()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    #@unittest.skip("not test")
    def test_all_reduce_product(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.PRODUCT,
            10,
            2,
            reduce((lambda x, y: x * y), [2] * (len(group) - 1), 10),
        )

    #@unittest.skip("not test")
    def test_all_reduce_min(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MIN,
            1,
            1010,
            1,
        )

    #@unittest.skip("not test")
    def test_all_reduce_max(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MAX,
            10,
            -1,
            10,
        )

    def _test_all_gather_helper(self, group, rank, times=1):
        def _build_tensor(size, value):
            return torch.arange(value, size*size*size + value).view(size, size, size).float()
        ct.set_device(rank % ct.device_count())
        loop_list = list(range(times))
        dtype_list = [("torch.FloatTensor", True),
                      ("torch.HalfTensor", True),
                      ("torch.CharTensor", True),
                      ("torch.ByteTensor", True),
                      ("torch.IntTensor", True),
                      ("torch.LongTensor", True),
                      ("torch.DoubleTensor", True)]
        list_list = [loop_list, dtype_list, group]
        for _, dtype_tuple, src in product(*list_list):
            if not dtype_tuple[1]:
                continue
            ttype = dtype_tuple[0]
            tensor = self.to_device(_build_tensor(src + 1, rank).type(ttype))
            tensors = [self.to_device(_build_tensor(src + 1, -1).type(ttype))
                        for i in group]
            dist.all_gather(tensors, tensor)
            expected_tensors = [self.to_device(_build_tensor(src + 1, i).type(ttype))
                                for i in group]
            for t1, t2 in zip(tensors, expected_tensors):
                self.assertTrue(t1.eq(t2).cpu().min().item())

    #@unittest.skip("not test")
    def test_all_gather(self):
        group, rank = self._init_global_test()
        self._test_all_gather_helper(group, rank)

    #@unittest.skip("not test")
    def test_pressure(self):
        group, rank = self._init_global_test()
        self._test_all_gather_helper(group, rank, times=20)

    def _test_p2pop_helper(self, rank):
        dtype_list = ["torch.FloatTensor", "torch.HalfTensor", "torch.CharTensor",
                      "torch.ByteTensor", "torch.IntTensor", "torch.LongTensor",
                      "torch.DoubleTensor"]
        ct.set_device(rank % ct.device_count())
        for ttype in dtype_list:
            send_tensor = torch.tensor(range(10)).type(ttype).to("mlu")
            recv_tensor = torch.zeros(10).type(ttype).to("mlu")
            p2p_op_list = []
            if rank == 0:
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, 1))
            elif rank == 1:
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, 0))
            if rank in [0, 1]:
                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()
            if rank == 1:
                self.assertTensorsEqual(recv_tensor.float().cpu(), send_tensor.float().cpu(), 0)

    #@unittest.skip("not test")
    def test_p2pop(self):
        os.environ["CNCL_SEND_RECV_ENABLE"] = str(1)
        _, rank = self._init_global_test()
        self._test_p2pop_helper(rank)

    #@unittest.skip("not test")
    def test_batch_isend_irecv(self):
        os.environ["CNCL_SEND_RECV_ENABLE"] = str(1)
        _, rank = self._init_global_test()
        ct.set_device(rank % ct.device_count())

        for val in ["0", "1"]:
            p2p_op_list = []
            os.environ["CNCL_BLOCKING_WAIT"] = val
            cncl_group = dist.new_group(backend="cncl")
            for src in range(0, dist.get_world_size()):
                if src == rank:
                    send_tensor = _build_tensor(rank + 1, value=-1).to("mlu")
                    for dst in range(0, dist.get_world_size()):
                        if dst != rank:
                            send_op = dist.P2POp(dist.isend, send_tensor, dst, group=cncl_group)
                            p2p_op_list.append(send_op)
                else:
                    recv_tensor = _build_tensor(src + 1).to("mlu")
                    recv_op = dist.P2POp(dist.irecv, recv_tensor, src, group=cncl_group)
                    p2p_op_list.append(recv_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()
            dist.destroy_process_group(cncl_group)

    #@unittest.skip("not test")
    def test_batch_isend_irecv_cncl_self(self):
        os.environ["CNCL_SEND_RECV_ENABLE"] = str(1)
        _, rank = self._init_global_test()
        ct.set_device(rank % ct.device_count())
        p2p_op_list = []

        if rank == 0:
            send_tensor = _build_tensor(1, value=-1).to("mlu")
            recv_tensor = _build_tensor(1).to("mlu")
            send_op = dist.P2POp(dist.isend, send_tensor, 0)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
            p2p_op_list.append(send_op)
            p2p_op_list.append(recv_op)

            with self.assertRaisesRegex(RuntimeError,
                "Cncl does not support p2p commucation on one deivce"):
                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

    def _test_barrier_helper(self, group, rank):
        ct.set_device(rank % ct.device_count())
        WAIT_TIME = 10  # seconds

        # Because MLU does not support Double currently, the precision of the float cast result
        # of time.time() is not enough, so we remainder the value by 100000
        for src in group:
            expected_time = self.to_device(torch.FloatTensor(1).fill_(0.0))
            if src == rank:
                expected_time.fill_(time.time() % 100000 + WAIT_TIME)
                dist.broadcast(expected_time, src)
                time.sleep(WAIT_TIME)
                dist.barrier()
            else:
                dist.broadcast(expected_time, src)
                dist.barrier()
                finish_time = time.time() % 100000
                self.assertGreaterEqual(float(finish_time), float(expected_time.item()),
                  "destination rank: %d, my rank: %d" % (src, rank))

    #@unittest.skip("not test")
    def test_barrier(self):
        group, rank = self._init_global_test()
        self._test_barrier_helper(group, rank)

    @classmethod
    def _model_step(cls, model):
        for param in model.parameters():
            if param.grad is not None:
                param.data = param.data + param.grad
                param.grad.detach_()
                param.grad.zero_()

    def _prepare_dummy_data(self, local_bs):
        # global_bs for DDP should be divisible by WORLD_SIZE
        global_bs = self.args.nproc_per_node * self.args.nnodes * local_bs
        input_cpu = torch.randn(global_bs, 2)
        target = torch.randn(global_bs, 4)
        loss = nn.MSELoss()
        return global_bs, input_cpu, target, loss

    # END TO END TEST FOR DISTRIBUTEDDATAPARALLEL
    @classmethod
    def _test_DDP_helper(cls, model, input_var, target, loss, flg):
        model.train()
        output = model(input_var, flg)
        l = loss(output, target)
        l.backward()

    def _assert_equal_param(self, param, param_DDP):
        self.assertEqual(len(param), len(param_DDP))
        for p, p_DDP in zip(param, param_DDP):
            self.assertTensorsEqual(p, p_DDP.cpu(), 3e-3)

    def _test_multi_nodes_helper(self, param_DDP, rank):
        ps = []
        file_name = "params_" + str(self.world_size) + "cards.pt"
        single_node_params_file = os.path.join(TEMP_DIR, file_name)
        if  self.args.nnodes == 1:
            if rank == 0:
                for p in param_DDP:
                    ps.append(p.cpu())
                torch.save(ps, single_node_params_file)
        else:
            if os.path.exists(single_node_params_file):
                ps = torch.load(single_node_params_file, map_location = torch.device('cpu'))
                for p_sing, p_mult in zip(ps, param_DDP):
                    self.assertTensorsEqual(p_sing, p_mult.cpu(), 0)
            else:
                print("WARNING: " + single_node_params_file + " not found, if you want to "
                      "compare with single mlu card parameters of Net, please run single "
                      "node version of test_distributed.py firstly!")

    def _test_DDP_5iter(self, model_base, model_DDP, input_data, target,
                        loss, local_bs, rank, batch_size):
        for _ in range(5):
            # single cpu training
            self._test_DDP_helper(model_base, input_data, target, loss, False)

            # DDP training, DDP scatters subsets of input_cpu to nodes/MLUs
            self._test_DDP_helper(model_DDP, input_data[rank * local_bs: (rank + 1) * local_bs],
                target[rank * local_bs: (rank + 1) * local_bs], loss, True)

            # Update weights and run a second iteration to shake out errors
            self._model_step(model_base)
            self._model_step(model_DDP)
            self._assert_equal_param(list(model_base.parameters()),
                                     list(model_DDP.module.parameters()))

            # Shuffle the input so that DDP input is different
            input_data = input_data[torch.randperm(batch_size)]
        self._test_multi_nodes_helper(list(model_DDP.module.parameters()), rank)

    def _test_DistributedDataParallel(self, rank):
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        ct.set_device(rank % ct.device_count())

        # cpu training setup
        model = Net()
        #model.fc1.weight.register_hook(hook)

        # DDP training setup
        model_DDP = copy.deepcopy(model)
        model_DDP.to(ct.mlu_device())
        # can use find_unused_parameters=True
        model_DDP = nn.parallel.DistributedDataParallel(model_DDP,
            device_ids=[rank % ct.device_count()])
        def hook(grad):     # pylint: disable=W0612
            print("hook no_grad_param: ", model_DDP.module.no_grad_param.size(),
              model_DDP.module.no_grad_param.cpu())
            return grad
        #model_DDP.module.fc1.weight.register_hook(hook)

        # dummy data initialization
        local_bs = 1
        global_bs, input_cpu, target, loss = self._prepare_dummy_data(local_bs)

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model,
            model_DDP,
            input_cpu,
            target,
            loss,
            local_bs,
            rank,
            global_bs,
        )

    #@unittest.skip("not test")
    def test_distributedDataParallel(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)

    #@unittest.skip("not test")
    def test_abnormal_and_api(self):
        _, rank = self._init_global_test()
        ct.set_device(rank % ct.device_count())
        tensors = [self.to_device(torch.randn(2))]
        pg = _get_default_group()

        # test basic api
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)
        self.assertTrue(dist.is_initialized())

        # test unsupported communicate op
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.allgather_coalesced([tensors], tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.allreduce_coalesced(tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.reduce_scatter(tensors[0], tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.gather([tensors], tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.scatter(tensors, [tensors])
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.recv_anysource(tensors, 0)
        # use abnormal input tensors to test
        with self.assertRaisesRegex(RuntimeError, "Expecting all tensors on the same device"):
            pg.allgather([[tensors[0].cpu()]], tensors)
        with self.assertRaisesRegex(RuntimeError, "All tensor operands to scatter/gather " +
                                                  "must have the same size"):
            pg.allgather([tensors], [self.to_device(torch.randn(3))])
        with self.assertRaisesRegex(RuntimeError, "MLU Tensors must be on a single MLU " +
                                                  "device per process"):
            pg.allgather([tensors], [tensors[0], tensors[0]])
        with self.assertRaisesRegex(RuntimeError, "MLU Tensors must be on a single MLU " +
                                                  "device per process"):
            pg.allreduce([tensors[0], tensors[0]])
        with self.assertRaisesRegex(RuntimeError, "Unsupported data type for CNCL process group"):
            pg.allreduce(tensors[0].bool())
        with self.assertRaisesRegex(RuntimeError, "Tensor list must be nonempty"):
            pg.broadcast([])
        with self.assertRaisesRegex(
                RuntimeError, "Tensor list mustn't be larger than the number of available MLUs"):
            exceed_tensor_list = [tensors[0]
                                  for _ in range(ct.device_count() + 1)]
            pg.broadcast(exceed_tensor_list)
        with self.assertRaisesRegex(RuntimeError, "Tensors must be MLU and dense"):
            pg.broadcast([tensors[0].cpu()])

        dist.destroy_process_group()
        self.assertFalse(dist.is_initialized())
        pg = None

class TestDistBackend(TestCase, _DistTestBase):
    MANAGER_PROCESS_RANK = -1
    sync_manager = None

    @staticmethod
    def manager_join(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MANAGER_PROCESS_RANK:
                self._join_and_reduce()  # pylint: disable=W0212
            else:
                fn(self)

        return wrapper

    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = cls.args.master_addr
        os.environ["MASTER_PORT"] = str(cls.args.master_port)
        for attr in dir(cls):
            if attr.startswith("test"):
                fn = getattr(cls, attr)
                if not getattr(fn, "__unittest_skip__", False):
                    setattr(cls, attr, cls.manager_join(fn))
        if cls.args.node_rank == 0:
            queue = Queue()
            QueueManager.register(str("get_queue"), lambda:queue)
            cls.sync_manager = QueueManager(address=("", cls.args.master_port + 10),
                                            authkey=b'abc')
            cls.sync_manager.start()
        else:
            QueueManager.register(str("get_queue"))
            cls.sync_manager = QueueManager(address=(cls.args.master_addr,
                                                     cls.args.master_port + 10), authkey=b'abc')

    @classmethod
    def tearDownClass(cls):
        if cls.args.node_rank == 0:
            queue = cls.sync_manager.get_queue()
            while queue.empty() is False:
                time.sleep(0.1)
            cls.sync_manager.shutdown()

    def setUp(self):
        super(TestDistBackend, self).setUp()   # pylint: disable=R1725
        self.processes = []
        self.rank = self.MANAGER_PROCESS_RANK
        for rank in range(self.args.node_rank * self.args.nproc_per_node,
                self.args.node_rank * self.args.nproc_per_node + self.args.nproc_per_node):
            self.processes.append(self._spawn_process(rank))

    def tearDown(self):
        super(TestDistBackend, self).tearDown()   # pylint: disable=R1725

        for p in self.processes:
            p.terminate()

    def _spawn_process(self, rank):
        os.environ["RANK"] = str(rank)
        name = "process " + str(rank)
        process = multiprocessing.Process(
            target=self._run, name=name, args=(rank, ))
        process.start()
        return process

    def _barrier(self, rank):
        self.sync_manager.connect()
        q = self.sync_manager.get_queue()
        if rank == 0:
            for _ in range(self.world_size - 1):
                q.put(str(os.getpid()))
        else:
            print("received finish signal from Process", q.get())

    def _run(self, rank):
        if ct.device_count() < args.nproc_per_node:
            print("Lack MLU Device !!!!!!")
            sys.exit(0)

        self.rank = rank
        self.world_size = self.args.nproc_per_node * self.args.nnodes
        try:
            print("begin init Process", os.getpid())
            dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                world_size=self.world_size, rank=self.rank)
            print("end init Process", os.getpid())
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(SKIP_IF_BACKEND_UNAVAILABLE)
            raise

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, self.id().split(".")[2])()

        # must close the current listenning socket before doing barrier,
        # otherwise the connecting request of the pg of the next test
        # case might be listened
        if dist.is_initialized():
            dist.destroy_process_group()
        self._barrier(rank)

        sys.exit(0)

    def _join_and_reduce(self):
        join_timeout = get_timeout(self.id()) + self.args.delay_time
        for rank, process in enumerate(self.processes):
            process.join(join_timeout)
            self.assertFalse(process.is_alive(),
                             "Timeout waiting for rank %d to terminate" % rank)

        first_process = self.processes[0]
        for p in self.processes:
            self.assertEqual(p.exitcode, first_process.exitcode)

        if first_process.exitcode == SKIP_IF_BACKEND_UNAVAILABLE:
            raise unittest.SkipTest("Compiled without the cncl backend")

        self.assertEqual(first_process.exitcode, 0)

if __name__ == '__main__':
    args = parse_args()

    if args.nnodes > 1:
        args.connects = -1

    if args.connects != -1:
        os.environ["CNCL_SHM_DISABLE"] = "0"

    if args.connects > 0:
        os.environ["CNCL_C2C_DISABLE"] = "1"
    if args.connects > 1:
        os.environ["CNCL_P2P_LEVEL"] = "0"
    if args.connects > 2:
        os.environ["CNCL_SHM_DISABLE"] = "1"

    distutils.dir_util.mkpath(TEMP_DIR)

    _DistTestBase.args = args
    unittest.main(argv=[sys.argv[0]] + args.unittest_args, verbosity=2)
