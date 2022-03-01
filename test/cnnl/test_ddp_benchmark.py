#!/usr/bin/env python3
#
# Measure distributed training iteration time.
#
# This program performs a sweep over a) a number of model architectures, and
# b) an increasing number of processes. This produces a 1-DEV baseline,
# an 8-DEV baseline (if applicable), as well as measurements for however
# many processes can participate in training.
#
# pylint: disable=W0511
import argparse
import itertools
import json
import os
import shlex
import subprocess
from subprocess import PIPE
import sys
import time
import pickle

import numpy as np
import torch
import torch.distributed as dist
# TODO(zhanchendi): can be removed when update pytorch
from torch.distributed.distributed_c10d import _rank_not_in_group
import torch.nn as nn
import torch.optim as optim
import torchvision


if not torch._six.PY3:
    raise RuntimeError("DDP benchmark requires Python 3")

# TODO(zhanchendi): these function can be removed when update pytorch
def _object_to_tensor(obj):
    buffer = pickle.dumps(obj)
    byte_storage = torch.ByteStorage.from_buffer(buffer)
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size

# TODO(zhanchendi): these function can be removed when update pytorch
def _tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    out = pickle.loads(buf)
    return out

# TODO(zhanchendi): all_gather_object is supported by higher version of pytorch, but no pytorch1.6
# these function can be removed when update pytorch
def all_gather_object(object_list, obj, group=dist.group.WORLD):
    """
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.
    Arguments:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        object (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on
    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.
    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.
    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.
    """
    if _rank_not_in_group(group):
        return

    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = dist.get_backend(group)
    my_rank = dist.get_rank()
    is_nccl_backend = group_backend == dist.Backend.NCCL
    if is_nccl_backend:
        input_tensor, local_size = input_tensor.to(my_rank), local_size.to(my_rank)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = dist.get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=int).to(
        my_rank if is_nccl_backend else "cpu"
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = max(object_size_list)
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8
    ).to(my_rank if is_nccl_backend else "cpu")
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    dist.all_gather(output_tensors, input_tensor, group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.ByteTensor)
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)

def allgather_object(obj):
    out = [None for _ in range(dist.get_world_size())]
    all_gather_object(out, obj)
    return out

def allgather_run(cmd):
    proc = subprocess.run(shlex.split(cmd), check=True, stdout=PIPE, stderr=PIPE)
    assert proc.returncode == 0
    return allgather_object(proc.stdout.decode("utf-8"))

def allequal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def benchmark_process_group(pg, benchmark, use_ddp_for_single_rank=True):
    torch.manual_seed(pg.rank())
    if benchmark.distributed_backend != 'cncl':
        torch.cuda.manual_seed(pg.rank())
    else:
        import torch_mlu.core.mlu_model as ct    # pylint: disable=C0415

    model = benchmark.create_model()
    data = [(benchmark.generate_inputs(), benchmark.generate_target())]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        0.001,
        momentum=0.9,
        weight_decay=1e-4)
    if use_ddp_for_single_rank or pg.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()
                        if benchmark.distributed_backend != 'cncl' else ct.current_device()],
            broadcast_buffers=False,
            process_group=pg,
            bucket_cap_mb=benchmark.bucket_size)

    measurements = []
    warmup_iterations = 5
    measured_iterations = 10
    for (inputs, target) in data * (warmup_iterations + measured_iterations):
        start = time.time()
        output = model(*inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if benchmark.distributed_backend == 'cncl':
            ct.synchronize()
        else:
            torch.cuda.synchronize()
        measurements.append(time.time() - start)

    # Throw away measurements for warmup iterations
    return measurements[warmup_iterations:]


def run_benchmark(benchmark, ranks, opts):
    group = dist.new_group(ranks=ranks, backend=benchmark.distributed_backend)
    measurements = []
    if dist.get_rank() in set(ranks):
        if not opts:
            opts = dict()
        measurements = benchmark_process_group(group, benchmark, **opts)
    dist.destroy_process_group(group)
    dist.barrier()

    # Aggregate measurements for better estimation of percentiles
    return list(itertools.chain(*allgather_object(measurements)))


def sweep(benchmark):
    # Synthesize the set of benchmarks to run.
    # This list contain tuples for ("string prefix", [rank...]).
    benchmarks = []

    def append_benchmark(prefix, ranks, opts=None):
        prefix = "%4d DEVs -- %s" % (len(ranks), prefix)
        benchmarks.append((prefix, ranks, opts))

    def local_print(msg):
        if dist.get_rank() == 0:
            print(msg, end='', flush=True)  # noqa: E999

    def print_header():
        local_print("\n")
        local_print("%22s" % "")
        for _ in [50, 75, 90, 95]:
            local_print("%14s%10s" % ("sec/iter", "ex/sec"))
        local_print("\n")

    def print_measurements(prefix, nelem, measurements):
        measurements = sorted(measurements)
        local_print("%8s:" % prefix)
        for p in [50, 75, 90, 95]:
            v = np.percentile(measurements, p)
            local_print("  p%02d:  %1.3fs  %6d/s" % (p, v, nelem / v))
        local_print("\n")

    # Every process runs once by themselves to warm up (CUDA init, etc).
    append_benchmark("  warmup", [dist.get_rank()], {"use_ddp_for_single_rank": False})

    # Single machine baselines
    append_benchmark("  no ddp", range(1), {"use_ddp_for_single_rank": False})
    append_benchmark("   1M/1G", range(1))
    append_benchmark("   1M/2G", range(2))
    append_benchmark("   1M/4G", range(4))

    # Multi-machine benchmarks
    for i in range(1, (dist.get_world_size() // 8) + 1):
        append_benchmark("   %dM/8G" % i, range(i * 8))

    # Run benchmarks in order of increasing number of DEVs
    print_header()
    results = []
    for prefix, ranks, opts in sorted(benchmarks, key=lambda tup: len(tup[1])):
        # Turn range into materialized list.
        ranks = list(ranks)
        measurements = run_benchmark(benchmark, ranks, opts)
        if "warmup" not in prefix:
            print_measurements(prefix, benchmark.batch_size, measurements)
            results.append({"ranks": ranks, "measurements": measurements})

    return results


class Benchmark(object):
    def __init__(self, device, distributed_backend, bucket_size):
        self.device = device
        self.batch_size = 32
        self.distributed_backend = distributed_backend
        self.bucket_size = bucket_size

    def __str__(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def generate_inputs(self):
        raise NotImplementedError

    def generate_target(self):
        raise NotImplementedError


class TorchvisionBenchmark(Benchmark):
    def __init__(self, device, distributed_backend, bucket_size, model):
        super(TorchvisionBenchmark, self).__init__(
            device,
            distributed_backend,
            bucket_size,
        )
        self.model = model

    def __str__(self):
        return "{} with batch size {}".format(self.model, self.batch_size)

    def create_model(self):
        return torchvision.models.__dict__[self.model]().to(self.device)

    def generate_inputs(self):
        return [torch.rand([self.batch_size, 3, 224, 224]).to(self.device)]

    def generate_target(self):
        return torch.tensor([1] * self.batch_size, dtype=torch.long).to(self.device)


def main():  # pylint: disable=R0912
    parser = argparse.ArgumentParser(description='PyTorch distributed benchmark suite')
    parser.add_argument("--rank", type=int, default=os.environ["RANK"])
    parser.add_argument("--world-size", type=int, default=os.environ["WORLD_SIZE"])
    parser.add_argument("--distributed-backend", type=str, default="cncl")
    parser.add_argument("--bucket-size", type=int, default=25)
    parser.add_argument("--master-addr", type=str, default=os.environ["MASTER_ADDR"])
    parser.add_argument("--master-port", type=str, default=os.environ["MASTER_PORT"])
    parser.add_argument("--model", type=str)
    parser.add_argument("--json", type=str, metavar="PATH",
                        help="Write file with benchmark results")
    args = parser.parse_args()

    # The global process group used only for communicating benchmark
    # metadata, like measurements. Not for benchmarking itself.
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
        rank=args.rank,
        world_size=args.world_size,
    )

    if args.distributed_backend == "cncl":
        cmd = "cnmon topo"
    else:
        cmd = "nvidia-smi topo -m"
    output = allgather_run(cmd)
    if not allequal(output):
        print('Output of "', cmd, '" differs between machines')
        sys.exit(1)

    if args.rank == 0:
        print("-----------------------------------")
        print("PyTorch distributed benchmark suite")
        print("-----------------------------------")
        print("")
        print("* PyTorch version: {}".format(torch.__version__))
        if args.distributed_backend != "cncl":
            print("* CUDA version: {}".format(torch.version.cuda))
        print("* Distributed backend: {}".format(args.distributed_backend))
        print("* Maximum bucket size: {}MB".format(args.bucket_size))
        print("")
        print("--- ", cmd," ---")
        print("")
        print(output[0])
        print("--------------------------")
        print("")

    if args.distributed_backend == "cncl":
        import torch_mlu.core.mlu_model as ct    # pylint: disable=C0415
        ct.set_device(dist.get_rank() % 8)
        device = torch.device('mlu')
    else:
        torch.cuda.set_device(dist.get_rank() % 8)
        device = torch.device('cuda:%d' % (dist.get_rank() % 8))

    benchmarks = []
    if args.model:
        benchmarks.append(
            TorchvisionBenchmark(
                device=device,
                distributed_backend=args.distributed_backend,
                bucket_size=args.bucket_size,
                model=args.model))
    else:
        for model in ["resnet50", "resnet101", "resnext50_32x4d"]: # "resnext101_32x8d"]:
            benchmarks.append(
                TorchvisionBenchmark(
                    device=device,
                    distributed_backend=args.distributed_backend,
                    bucket_size=args.bucket_size,
                    model=model))

    benchmark_results = []
    for benchmark in benchmarks:
        if args.rank == 0:
            print("\nBenchmark: {}".format(str(benchmark)))
        result = sweep(benchmark)
        benchmark_results.append({
            "model": benchmark.model,
            "batch_size": benchmark.batch_size,
            "result": result,
        })

    # Write file with benchmark results if applicable
    if args.rank == 0 and args.json:
        report = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "distributed_backend": args.distributed_backend,
            "bucket_size": args.bucket_size,
            "benchmark_results": benchmark_results,
        }
        with open(args.json, 'w') as f:
            json.dump(report, f)


if __name__ == '__main__':
    main()
