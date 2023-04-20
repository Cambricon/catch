import os
import sys
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import find_free_port
from torchvision.datasets import FakeData
import torchvision.transforms as transforms
import torchvision.models as models
import torch_mlu.core.mlu_model as ct

INIT_METHOD = 'tcp://127.0.0.5:' + str(find_free_port())
TIMEOUT = 100

def spawn_processes(world_size, func):
    processes = []
    # start task
    for rank in range(world_size):
        name = "process " + str(rank)
        process = mp.Process(target=func, name=name, args=(rank, world_size))
        process.start()
        processes.append(process)
    # wait task completion
    for rank, process in enumerate(processes):
        process.join(TIMEOUT)
        if process.is_alive():
            print("Timeout waiting for rank %d to terminate" % rank)
            process.terminate()

def do_broadcast(rank, world_size):
    ct.set_device(rank)  # assign a card for every process, rank is the card id

    # initialize the process group
    dist.init_process_group(backend='cncl', init_method=INIT_METHOD, # 此处的backend是否应是nccl？
                            rank=rank, world_size=world_size)

    # broadcast operation
    t = torch.randn(10).to("mlu")
    src = 0  # broadcast from card 0 to other cards
    dist.broadcast(t, src)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)

def do_allreduce(rank, world_size):
    ct.set_device(rank)  # assign a card for every process, rank is the card id

    # initialize the process group
    dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                            rank=rank, world_size=world_size)

    t = torch.randn(10).to("mlu")
    # allreduce operation, support sum/product/min/max operation
    dist.all_reduce(t, dist.ReduceOp.SUM)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)

def do_send_recv(rank, world_size):
    os.environ["CNCL_SEND_RECV_ENABLE"] = str(1)
    ct.set_device(rank)  # assign a card for every process, rank is the card id

    # initialize the process group
    dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                            rank=rank, world_size=world_size)

    p2p_op_list = []
    if rank == 0:
        send_tensor = torch.randn(10).to("mlu:{}".format(rank))
        send_op = dist.P2POp(dist.isend, send_tensor, 1)
        p2p_op_list.append(send_op)
    elif rank == 1:
        recv_tensor = torch.randn(10).to("mlu:{}".format(rank))
        recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
        p2p_op_list.append(recv_op)

    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()
    ct.synchronize()

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)

def do_reduce(rank, world_size):
    ct.set_device(rank)  # assign a card for every process, rank is the card id

    # initialize the process group
    dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                            rank=rank, world_size=world_size)

    t = torch.randn(10).to("mlu")
    # allreduce operation, support sum/product/min/max operation
    dist.reduce(t, 0, dist.ReduceOp.SUM)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)

def do_allgather(rank, world_size):
    ct.set_device(rank)  # assign a card for every process, rank is the card id

    # initialize the process group
    dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                            rank=rank, world_size=world_size)

    t = torch.randn(10).to("mlu")
    ts = [torch.randn(10).to("mlu") for i in range(2)]
    # allgather operation
    dist.all_gather(ts, t)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)

def do_barrier(rank, world_size):
    ct.set_device(rank)  # assign a card for every process, rank is the card id

    # initialize the process group
    dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                            rank=rank, world_size=world_size)

    # barrier operation
    dist.barrier()

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)

BATCH_SIZE = 16
LR = 0.01
MOMENTUM = 0.1
WEIGHT_DECAY = 0.1
def multi_card_train(rank, world_size):
    ct.set_device(rank)  # assign a card for every process, rank is the card id

    # initialize the process group
    dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                            rank=rank, world_size=world_size)

    # init dataloader
    train_dataset = FakeData(size=BATCH_SIZE * 6, transform=transforms.ToTensor())
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=2)

    # init DDP model
    model = models.resnet50()
    model.to(ct.mlu_device())  # copy model weights to MLU device
    model = DDP(model, device_ids=[rank])
    model.train()
    criterion = nn.CrossEntropyLoss()
    criterion.to(ct.mlu_device())
    optimizer = torch.optim.SGD(model.parameters(), LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY)
    ct.to(optimizer, torch.device('mlu'))

    for _, (images, target) in enumerate(train_loader):
        images = images.to("mlu")
        target = target.to("mlu")
        output = model(images)  # forward propagation
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()  # backward propagation
        optimizer.step()

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)

def main():
    world_size = 2 # set card number
    # collective operations based on MLU cards
    spawn_processes(world_size, do_broadcast)
    spawn_processes(world_size, do_allreduce)
    spawn_processes(world_size, do_reduce)
    spawn_processes(world_size, do_allgather)
    spawn_processes(world_size, do_send_recv)
    spawn_processes(world_size, do_barrier)
    # distributed train based on MLU cards
    spawn_processes(world_size, multi_card_train)

if __name__ == "__main__":
    main()
