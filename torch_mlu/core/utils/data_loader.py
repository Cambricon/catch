from __future__ import print_function

import os
import warnings
import torch
import torch.utils.data.dataloader as dataloader
import torch.utils.data._utils as _utils
import torch.utils.data._utils.worker as worker
import torch.utils.data._utils.pin_memory as pin_memory
import torch.multiprocessing as multiprocessing
import threading
import itertools
from torch._six import queue, container_abcs, string_classes
from torch._utils import ExceptionWrapper
from torch.utils.data.dataloader import _DatasetKind
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.device.notifier as Notifier


def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    ct.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory.pin_memory(data)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        del r  # save memory

class _MLUMultiProcessingDataLoaderIter(dataloader._MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        dataloader._BaseDataLoaderIter.__init__(self, loader)

        assert self._num_workers > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue()
        self._worker_pids_set = False
        self._shutdown = False
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        self._workers_status = []
        self._use_io_queue = bool((os.getenv('USE_IO_QUEUE') is not None) 
                                  and (os.getenv('USE_IO_QUEUE').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']))
        if self._use_io_queue:
            warnings.warn("using io_queue in data_loader will take more memory on MLU.")
            self._mlu_data = None
            self._notifier = Notifier.Notifier()
            self._io_queue = torch_mlu._MLUC._getQueueFromPool(-1)
            self._current_queue = None
            self._mlu_rcvd_idx = 0 # idx of the next task to be returned in __next__ when using io queue 

        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            # index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed + i, self._worker_init_fn, i, self._num_workers))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
            self._workers_status.append(True)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._data_queue = queue.Queue()
            pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      ct.current_device(),
                      self._pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True

        # prime the prefetch loop
        for _ in range(2 * self._num_workers):
            self._try_put_index()
       
        if self._use_io_queue:
            self._mlu_data = self._next_mlu_data()

    def _next_mlu_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            data_to_mlu = None
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._data_to_mlu(self._process_data(data))

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1

            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._data_to_mlu(self._process_data(data))

    def _data_to_mlu(self, data):
        self._current_queue = torch_mlu._MLUC._getCurrentQueue(-1)
        torch_mlu._MLUC._setCurrentQueue(self._io_queue)
        ct.set_memory_strategy(True)
        data = self._to_mlu(data)
        self._notifier.place()
        ct.set_memory_strategy(False)
        torch_mlu._MLUC._setCurrentQueue(self._current_queue)
        return data

    def _to_mlu(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("mlu", non_blocking=True)
        elif isinstance(data, string_classes):
            return data
        elif isinstance(data, container_abcs.Mapping):
            return {k: self._to_mlu(sample) for k, sample in data.items()}
        elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
            return type(data)(*(self._to_mlu(sample) for sample in data))
        elif isinstance(data, container_abcs.Sequence):
            return [self._to_mlu(sample) for sample in data]
        elif hasattr(data, "to"):
            return data.to("mlu", non_blocking=True)
        else:
            return data

    def _next_data(self):
        if self._use_io_queue:
            self._notifier.wait()
            return_data = self._mlu_data
            if self._rcvd_idx < self._send_idx:
                self._mlu_data = self._next_mlu_data()
            else:
                self._shutdown_workers()
            if self._mlu_rcvd_idx >= self._rcvd_idx:
                raise StopIteration 
            self._mlu_rcvd_idx += 1
            return return_data
        else:
            return super()._next_data()
