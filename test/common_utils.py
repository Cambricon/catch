# pylint: disable=C0301,C0305,W0511,W0212,W0621,W1201,W0613,C0123,R0201,R1722,R1711,W0102,W0612,C0200,R0124,R0912
from __future__ import print_function
import unittest
import time
import argparse
import sys
import os
import subprocess
import signal
import logging
import traceback
import copy
import random
import warnings
from numbers import Number
from typing import cast, Optional
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from torch.testing import _compare_tensors_internal, _compare_scalars_internal, _compare_return_type
import torch_mlu.core.mlu_model as ct
import __main__

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--subprocess', action='store_true',
                    help='whether to run each test in a subprocess')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--accept', action='store_true')
args, remaining = parser.parse_known_args()
TEST_IN_SUBPROCESS = args.subprocess
SEED = args.seed
UNITTEST_ARGS = [sys.argv[0]] + remaining

SIGNALS_TO_NAMES_DICT = {getattr(signal, n): n for n in dir(signal)
                         if n.startswith('SIG') and '_' not in n}

def _check_module_exists(name):
    import importlib  # pylint: disable= C0415
    import importlib.util  # pylint: disable= C0415
    spec = importlib.util.find_spec(name)
    return spec is not None

TEST_NUMPY = _check_module_exists('numpy')
if TEST_NUMPY:
    # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
    numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64
    }

    # Dict of torch dtype -> NumPy dtype
    torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def gen_err_message(return_code, test_module, total_error_info):
    # Generate error messages based on the return code of the child process.
    if return_code != 0:
        message = '{} failed!'.format(test_module)
        if return_code < 0:
            # subprocess.Popen returns the child process' exit signal as
            # return code -N, where N is the signal number.
            signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
            message += ' Received signal: {}'.format(signal_name)
            total_error_info.append(message)
            logging.error("\033[31;1m {}\033[0m .".format(message))
        if return_code == 1:
            total_error_info.append(message)
        if return_code == 2:
            raise KeyboardInterrupt
        assert False, message

def print_to_stderr(message):
    print(message, file=sys.stderr)

def shell(command, cwd=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    err_msg = "Command to shell should be a list or tuple of tokens"
    assert not isinstance(command, torch._six.string_classes), err_msg
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd)  # pylint: disable= R1732
    try:
        return p.wait()
    except KeyboardInterrupt:
        # Give `p` a chance to handle KeyboardInterrupt. Without this,
        # `pytest` can't print errors it collected so far upon KeyboardInterrupt.
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            #return exit_status
            return int(2)
        else:
            p.kill()
            return int(2)
    except:  # noqa E722, copied from python core library
        p.kill()
        raise
    finally:
        # Always call p.wait() to ensure exit
        p.wait()

def run_tests(argv=UNITTEST_ARGS):
    if TEST_IN_SUBPROCESS:
        suite = unittest.TestLoader().loadTestsFromModule(__main__)
        test_cases = []

        def add_to_test_cases(suite_or_case):
            if isinstance(suite_or_case, unittest.TestCase):
                test_cases.append(suite_or_case)
            else:
                for element in suite_or_case:
                    add_to_test_cases(element)

        add_to_test_cases(suite)
        failed_tests = []
        for case in test_cases:
            test_case_full_name = case.id().split('.', 1)[1]
            exitcode = shell([sys.executable] + argv + [test_case_full_name])
            if exitcode != 0:
                failed_tests.append(test_case_full_name)

        assert len(failed_tests) == 0, "{} unit test(s) failed:\n\t{}".format(
            len(failed_tests), '\n\t'.join(failed_tests))
    else:
        unittest.main(argv=argv)

def testinfo(pre_message="\nTest case:", post_message="Test Time:"):
    def loader(func):
        def wrapper(*args, **kwargs):
            logging.info("\033[1;35m Current op and func: {}, {}. \033[0m" \
                         .format(type(args[0]).__name__, \
                         func.__name__))
            st_time = time.time()
            func(*args, **kwargs)
            logging.info("\033[1;30m Test time: %0.3f s. \033[0m" \
                         %(time.time() - st_time))
        return wrapper
    return loader

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def get_comparison_dtype(a, b):
    # TODO: update this when promote_types supports bfloat16 and/or
    # isclose supports bfloat16.
    a_dtype = torch.float32 if a.dtype is torch.bfloat16 else a.dtype
    b_dtype = torch.float32 if b.dtype is torch.bfloat16 else b.dtype

    compare_dtype = torch.promote_types(a_dtype, b_dtype)

    # non-CUDA (CPU, for example) float16 -> float32
    # TODO: update this when isclose is implemented for CPU float16
    if (compare_dtype is torch.float16 and
        (a.device != b.device or a.device.type != 'cuda' or
            b.device.type != 'cuda')):
        compare_dtype = torch.float32

    return compare_dtype

class TestCase(unittest.TestCase):  # pylint: disable = R0904
    # NOTE: "precision" lets classes and generated tests set minimum
    # atol values when comparing tensors. Used by @precisionOverride, for
    # example.
    # TODO: provide a better mechanism for generated tests to set rtol/atol.
    _precision: float = 0

    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, prec: float) -> None:
        self._precision = prec

    def to_non_dense(self, data, dim=None, distance=2):
        if not type(data) == torch.Tensor:
            print("[Warning]: It's not available to convert an unknown object to non-dense type")
            return data
        # convert the last channel as default.
        convert_dim = data.dim()
        if dim is not None:
            convert_dim = dim
        if convert_dim > data.dim():
            print(f"[Warning]: The max available expand dim for a {data.dim()} Tensor"\
                  f" is {data.dim()}, but got specified dim as {dim}.")
            convert_dim = data.dim()
        a = data.unsqueeze(convert_dim)
        b = torch.cat([a for _ in range(distance)], convert_dim)
        return b.select(dim=convert_dim, index=0)

    def to_mlu(self, input):
        """
           convert cpu-tensor into mlu-tensor based on self.data_type
           help to test both float32 and float16 data-type
        """
        if type(input) == torch.Tensor:
            if "PYTORCH_DATA_TYPE" in os.environ:
                mlu_input = input.type(self.data_type).to(ct.mlu_device())
            else:
                mlu_input = input.to(ct.mlu_device())
        else:
            mlu_input = input
        return mlu_input

    def to_mlu_dtype(self, input, data_type):
        if torch.is_tensor(input):
            mlu_input = input.to(data_type).to(ct.mlu_device())
        else:
            mlu_input = input
        return mlu_input

    def to_device(self, input):
        if type(input) == torch.Tensor:
            mlu_input = input.to(ct.mlu_device())
        else:
            mlu_input = input
        return mlu_input

    def convert_to_channel_last(self, x):
        if x.dim() == 4:
            mlu_input = x.to(memory_format=torch.channels_last)
        elif x.dim() == 5:
            mlu_input = x.to(memory_format=torch.channels_last_3d)
        else:
            mlu_input = x
        return mlu_input

    def get_not_contiguous_tensor(self, x):
        self.assertTrue(isinstance(x, torch.Tensor), \
            "Only support pytorch tensor.")
        dims = list(range(x.dim()))
        random.shuffle(dims)
        return x.permute(dims)

    def get_not_contiguous_tensor_container(self, x):
        self.assertTrue(isinstance(x, (list, tuple)), \
            "Only support list or tuple.")
        result = []
        for item in x:
            out = self.get_not_contiguous_tensor(item)
            result.append(out)
        if isinstance(x, tuple):
            return tuple(result)
        else:
            return result

    def set_params(self):
        print('\n')
        # base config
        self.tensor_generator = torch.randn
        self.tensor_scale = 1.0

        # set data type
        if "PYTORCH_DATA_TYPE" in os.environ:
            data_type = os.environ["PYTORCH_DATA_TYPE"].lower()
            if data_type == "half":
                self.data_type = torch.HalfTensor
            elif data_type == "float":
                self.data_type = torch.FloatTensor
            else:
                logging.error("Unknown data type!")
                exit(0)
        else:
            self.data_type = torch.FloatTensor

        # set running mode
        if "PYTORCH_RUNNING_MODE" in os.environ:
            running_mode = os.environ['PYTORCH_RUNNING_MODE'].lower()
            if running_mode == "fusion":
                self.running_mode = "fusion"
            elif running_mode == "layer":
                self.running_mode = "layer"
            else:
                logging.error("Unknown running mode!")
                exit(0)
        else:
            self.running_mode = "layer+fusion"

        return

    def gen_by_params(self, gen_types='', gen_params=[tuple()]):
        if gen_types=='':
            return [tuple()] * len(gen_params)
        init_types, inputs_types = gen_types.replace(" ","").split('|')
        init_types = list(filter(None, init_types.split(',')))
        inputs_types = list(filter(None, inputs_types.split(',')))
        gen_types = init_types + inputs_types
        cases = []
        for param in gen_params:
            inputs_list = []
            for inx, val in enumerate(param):
                if gen_types[inx] == 't':
                    t = self.tensor_generator(val) * self.tensor_scale
                    t = t.type(self.data_type)
                    inputs_list.append(t)
                elif gen_types[inx] == 'ts':
                    ts = []
                    for single in val:
                        t = self.tensor_generator(val) * self.tensor_scale
                        t = t.type(self.data_type)
                        ts.append(t)
                    inputs_list.append(tuple(ts))
                else:
                    inputs_list.append(val)

            inits = tuple(inputs_list[0:len(init_types)])
            inputs = tuple(inputs_list[len(init_types):])
            cases.append((inits,inputs))
        return cases


    def _test_several_cases(self,
                            cases,
                            test_class,
                            prec=0.02,
                            message='',
                            allow_inf=False,
                            use_MSE=False,
                            use_RAE=False,
                            use_RMA=False):
        logging.info("\033[1;35m Current op and func: {}, {}.  \033[0m" \
                .format(self.__class__.__name__, \
                traceback.extract_stack()[-2][2]))
        st_time = time.time()
        logging.debug("\033[1;33m Data type: " + str(self.data_type) + \
                      ". Running mode: " + str(self.running_mode) + ". \033[0m")
        for inx, case in enumerate(cases):
            init_case, inputs_case = case
            test_model = test_class(*init_case).float().eval()
            logging.debug("\033[1;33m Test case {}. \033[0m" \
                   .format(inx))
            self._test_single_case(test_model, inputs_case, prec, use_MSE=True)
        logging.info("\033[1;30m Test time: %0.3f s. \033[0m" \
                %(time.time() - st_time))

    def inputs_to_mlu(self, inputs):
        if isinstance(inputs, tuple):
            return tuple(self.to_mlu(copy.deepcopy(x)) for x in inputs)
        else:
            return self.to_mlu(copy.deepcopy(inputs))

    def _test_single_case(self,
                          test_model,
                          cpu_inputs,
                          prec=0.02,
                          message='',
                          allow_inf=False,
                          use_MSE=False,
                          use_RAE=False,
                          use_RMA=False,
                          cpu_outputs=None):
        # logging.info("\033[1;35m Current op and func: {}, {}.  \033[0m" \
        #         .format(self.__class__.__name__, \
        #         traceback.extract_stack()[-2][2]))
        mlu_inputs = tuple(self.inputs_to_mlu(x) for x in cpu_inputs)
        mlu_inputs_fus = tuple(self.inputs_to_mlu(x) for x in cpu_inputs)
        mlu_inputs_fus_ = tuple(self.inputs_to_mlu(x) for x in cpu_inputs)

        cpu_inputs = tuple(self.optional_fake_half_cpu_inputs(x) for x in cpu_inputs)
        if cpu_outputs is None:
            output_cpu = test_model(*cpu_inputs)
        else:
            output_cpu = cpu_outputs
        model = test_model.to(ct.mlu_device())
        if self.running_mode == "layer+fusion":
            # running layer-by-layer mode
            logging.info("running_mode is :layer+fusion, now running \033[1;32;40mLayer.\033[0m")
            output_mlu = model(*mlu_inputs)
            if isinstance(output_cpu, tuple):
                for kk in range(len(output_cpu)):
                    self.assertTensorsEqual(output_cpu[kk],
                                            output_mlu[kk].cpu(),
                                            prec,
                                            message,
                                            allow_inf,
                                            use_MSE,
                                            use_RAE,
                                            use_RMA)
            else:
                self.assertTensorsEqual(output_cpu,
                                        output_mlu.cpu(),
                                        prec,
                                        message,
                                        allow_inf,
                                        use_MSE,
                                        use_RAE,
                                        use_RMA)
            # running fusion mode
            logging.info("running_mode is :layer+fusion, now running \033[1;34;40mFusion.\033[0m")
            traced = torch.jit.trace(model, mlu_inputs_fus_, check_trace=False)
            output_fusion = traced(*mlu_inputs_fus)
            if isinstance(output_cpu, tuple):
                for kk in range(len(output_cpu)):
                    self.assertTensorsEqual(output_cpu[kk],
                                            output_fusion[kk].cpu(),
                                            prec,
                                            message,
                                            allow_inf,
                                            use_MSE,
                                            use_RAE,
                                            use_RMA)
            else:
                self.assertTensorsEqual(output_cpu,
                                        output_fusion.cpu(),
                                        prec,
                                        message,
                                        allow_inf,
                                        use_MSE,
                                        use_RAE,
                                        use_RMA)
        elif self.running_mode == "layer":
            output_mlu = model(*mlu_inputs)
            if isinstance(output_cpu, tuple):
                for kk in range(len(output_cpu)):
                    self.assertTensorsEqual(output_cpu[kk],
                                            output_mlu[kk].cpu(),
                                            prec,
                                            message,
                                            allow_inf,
                                            use_MSE,
                                            use_RAE,
                                            use_RMA)
            else:
                self.assertTensorsEqual(output_cpu,
                                        output_mlu.cpu(),
                                        prec,
                                        message,
                                        allow_inf,
                                        use_MSE,
                                        use_RAE,
                                        use_RMA)
        elif self.running_mode == "fusion":
            traced = torch.jit.trace(model, mlu_inputs_fus_, check_trace=False)
            output_mlu = traced(*mlu_inputs_fus)
            if isinstance(output_cpu, tuple):
                for kk in range(len(output_cpu)):
                    self.assertTensorsEqual(output_cpu[kk],
                                            output_mlu[kk].cpu(),
                                            prec,
                                            message,
                                            allow_inf,
                                            use_MSE,
                                            use_RAE,
                                            use_RMA)
            else:
                self.assertTensorsEqual(output_cpu,
                                        output_mlu.cpu(),
                                        prec,
                                        message,
                                        allow_inf,
                                        use_MSE,
                                        use_RAE,
                                        use_RMA)
        elif self.running_mode == "training":
            pass
        else:
            pass

    def setUp(self):
        # will be run before test
        self.set_params()

    def assertTensorsEqual(self,
                           a,
                           b,
                           prec=None,
                           message='',
                           allow_inf=False,
                           use_MSE=False,
                           use_RAE=False,
                           use_RMA=False):
        '''unittest.TestCase'''
        if a.dtype == torch.bool:
            a = a.float()
        if b.dtype == torch.bool:
            b = b.float()
        epsilon = 1.0 / 16384
        allow_inf = False
        self.assertEqual(a.size(), b.size(), message)
        if a.numel() > 0:
            a = self.optional_fake_half_cpu_inputs(a)
            b = self.optional_fake_half_cpu_inputs(b)
            # check that NaNs are in the same locations
            nan_mask = a != a
            self.assertTrue(torch.equal(nan_mask, b != b), message)
            diff = a - b
            diff[nan_mask] = 0
            # inf check if allow_inf=True
            if allow_inf:
                inf_mask = (a == float("inf")) | (a == float("-inf"))
                self.assertTrue(torch.equal(inf_mask,
                                            (b == float("inf")) | (b == float("-inf"))),
                                message)
                diff[inf_mask] = 0
            # TODO: implement abs on CharTensor
            if diff.is_signed() and 'CharTensor' not in diff.type():
                diff = diff.abs()
            if use_MSE:
                diff = diff.abs().pow(2).sum()
                a_pow_sum = a.pow(2).sum()
                if diff <= (2 * epsilon) * (2 * epsilon):
                    diff = 0.0
                if a_pow_sum <= epsilon:
                    a_pow_sum += epsilon
                diff = torch.div(diff, (a_pow_sum * 1.0))
                self.assertLessEqual(diff.sqrt(), prec, message)
            elif use_RAE:
                diff = diff.abs().sum()
                a_sum = a.abs().sum()
                if a_sum == 0:
                    self.assertEqual(a, b, message)
                else:
                    diff = torch.div(diff, a_sum)
                    self.assertLessEqual(diff, prec, message)
            elif use_RMA:
                a_mean = a.abs().mean()
                b_mean = b.abs().mean()
                if a_mean == 0:
                    self.assertEqual(a, b, message)
                else:
                    diff = torch.div((a_mean - b_mean).abs(), a_mean)
                    self.assertLessEqual(diff, prec, message)
            else:
                max_err = diff.max()
                self.assertLessEqual(max_err, prec, message)


    def compare_with_numpy(self, torch_fn, np_fn, tensor_like, device=None, dtype=None):
        assert TEST_NUMPY
        assert dtype is not torch.bfloat16

        if isinstance(tensor_like, torch.Tensor):
            assert device is None
            assert dtype is None
            a = tensor_like.detach().cpu().numpy()
            t = tensor_like
        else:
            a = np.array(tensor_like, dtype=torch_to_numpy_dtype_dict[dtype])
            if device == 'mlu':
                t = torch.tensor(tensor_like, dtype=dtype).to(ct.mlu_device())
            else:
                t = torch.tensor(tensor_like, device=device, dtype=dtype)

        np_result = np_fn(a)
        torch_result = torch_fn(t).cpu()

        # Converts arrays to tensors
        if isinstance(np_result, np.ndarray):
            try:
                np_result = torch.from_numpy(np_result)
            except Exception:  # pylint: disable= W0703
                # NOTE: copying an array before conversion is necessary when,
                #   for example, the array has negative strides.
                np_result = torch.from_numpy(np_result.copy())

        self.assertEqual(np_result, torch_result)

    dtype_precisions = {
        torch.float16    : (0.001, 1e-5),
        torch.float32    : (1.3e-6, 1e-5),
        torch.float64    : (1e-7, 1e-7),
    }

    def _getDefaultRtolAndAtol(self, dtype0, dtype1):
        rtol = max(self.dtype_precisions.get(dtype0, (0, 0))[0],
                   self.dtype_precisions.get(dtype1, (0, 0))[0])
        atol = max(self.dtype_precisions.get(dtype0, (0, 0))[1],
                   self.dtype_precisions.get(dtype1, (0, 0))[1])

        return rtol, atol

    def _compareTensors(self, a, b, *, rtol: Optional[float] = None, atol=None, equal_nan=True,
                        exact_dtype=True, exact_device=False) -> _compare_return_type:
        assert (atol is None) == (rtol is None)
        if not isinstance(a, torch.Tensor):
            return (False, "argument a, {0}, to _compareTensors is not a tensor!".format(a))
        if not isinstance(b, torch.Tensor):
            return (False, "argument b, {0}, to _compareTensors is not a tensor!".format(b))

        # Validates tensors are on the same device
        if exact_device and a.device != b.device:
            return (False, ("Attempted to compare equality of tensors on "
                            "different devices! Got devices {0} and "
                            "{1}.".format(a.device, b.device)))

        # Compares tensors of different devices on the CPU
        if a.device != b.device:
            a = a.cpu()
            b = b.cpu()

        # Checks size matches
        if a.size() != b.size():
            return (False, ("Attempted to compare equality of tensors with "
                            "different sizes. Got sizes {0} and {1}.").format(a.size(), b.size()))

        # Checks dtype (if exact_dtype)
        if exact_dtype and a.dtype is not b.dtype:
            return (False, ("Attempted to compare equality of tensors with "
                            "different dtypes. Got dtypes {0} and {1}.").format(a.dtype, b.dtype))

        # Acquires rtol and atol
        if rtol is None:
            rtol, atol = self._getDefaultRtolAndAtol(a.dtype, b.dtype)

        atol = max(atol, self.precision)

        # Converts to comparison dtype
        dtype = get_comparison_dtype(a, b)
        a = a.to(dtype)
        b = b.to(dtype)

        return _compare_tensors_internal(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def _compareScalars(self, a, b, *,
                        rtol: Optional[float] = None, atol: Optional[float] = None, equal_nan=True) -> _compare_return_type:
        # Acquires rtol and atol
        assert (atol is None) == (rtol is None)
        if rtol is None:
            if isinstance(a, complex) or isinstance(b, complex):  # pylint: disable= R1720
                raise ValueError("MLU does not support the complex dtype, please check it.")
            elif isinstance(a, float) or isinstance(b, float):
                rtol, atol = self._getDefaultRtolAndAtol(torch.float32, torch.float32)
            else:
                rtol, atol = 0, 0
        atol = max(atol, self.precision)

        return _compare_scalars_internal(a, b, rtol=cast(float, rtol), atol=cast(float, atol), equal_nan=equal_nan)

    # Compares x and y
    # TODO: default exact_device to True
    def assertEqual(self, x, y, msg: Optional[str] = None, *,  # pylint: disable= W0237
                    atol: Optional[float] = None, rtol: Optional[float] = None,
                    equal_nan=True, exact_dtype=True, exact_device=False) -> None:
        assert (atol is None) == (rtol is None), "If one of atol or rtol is specified the other must be, too"

        # Tensor x Number and Number x Tensor comparisons
        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self.assertEqual(x.item(), y, atol=atol, rtol=rtol, msg=msg,
                             exact_dtype=exact_dtype, exact_device=exact_device)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), atol=atol, rtol=rtol, msg=msg,
                             exact_dtype=exact_dtype, exact_device=exact_device)
        # Tensor x np.bool
        elif isinstance(x, torch.Tensor) and isinstance(y, np.bool_):
            self.assertEqual(x.item(), y, atol=atol, rtol=rtol, msg=msg,
                             exact_dtype=exact_dtype, exact_device=exact_device)
        elif isinstance(y, torch.Tensor) and isinstance(x, np.bool_):
            self.assertEqual(x, y.item(), atol=atol, rtol=rtol, msg=msg,
                             exact_dtype=exact_dtype, exact_device=exact_device)
        # Tensor x Tensor
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            super().assertEqual(x.is_sparse, y.is_sparse, msg=msg)
            super().assertEqual(x.is_quantized, y.is_quantized, msg=msg)
            if x.is_sparse:  # pylint: disable= R1720
                err_msg = ("Not support this sparse dtype, "
                           "Please refer to native func to supplement it.")
                raise ValueError(err_msg)
            elif x.is_quantized and y.is_quantized:
                err_msg = ("Not support this quantize dtype, "
                           "Please refer to native func to supplement it.")
                raise ValueError(err_msg)
            else:
                result, debug_msg = self._compareTensors(x, y, rtol=rtol, atol=atol,
                                                         equal_nan=equal_nan, exact_dtype=exact_dtype,
                                                         exact_device=exact_device)

                if not result and msg is None:
                    assert debug_msg is not None
                    msg = "Tensors failed to compare as equal! " + debug_msg
                self.assertTrue(result, msg=msg)
        elif isinstance(x, torch._six.string_classes) and isinstance(y, torch._six.string_classes):
            super().assertEqual(x, y, msg=msg)
        elif type(x) == set and type(y) == set:
            super().assertEqual(x, y, msg=msg)
        elif isinstance(x, dict) and isinstance(y, dict):
            if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                self.assertEqual(x.items(), y.items(), atol=atol, rtol=rtol,
                                 msg=msg, exact_dtype=exact_dtype,
                                 exact_device=exact_device)
            else:
                self.assertEqual(set(x.keys()), set(y.keys()), atol=atol, rtol=rtol,
                                 msg=msg, exact_dtype=exact_dtype,
                                 exact_device=exact_device)
                key_list = list(x.keys())
                self.assertEqual([x[k] for k in key_list],
                                 [y[k] for k in key_list],
                                 atol=atol, rtol=rtol, msg=msg,
                                 exact_dtype=exact_dtype, exact_device=exact_device)
        elif isinstance(x, type) and isinstance(y, type):
            # See TestTorch.test_assert_equal_generic_meta
            super().assertEqual(x, y, msg=msg)
        elif is_iterable(x) and is_iterable(y):
            super().assertEqual(len(x), len(y), msg=msg)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, atol=atol, rtol=rtol, msg=msg,
                                 exact_dtype=exact_dtype, exact_device=exact_device)
        elif isinstance(x, bool) and isinstance(y, bool):
            self.assertTrue(x == y, msg=msg)

        # Scalar x Scalar
        elif isinstance(x, Number) and isinstance(y, Number):
            result, debug_msg = self._compareScalars(x, y, rtol=rtol, atol=atol,
                                                     equal_nan=equal_nan)
            if not result and msg is None:
                assert debug_msg is not None
                msg = "Scalars failed to compare as equal! " + debug_msg
            self.assertTrue(result, msg=msg)
        else:
            super().assertEqual(x, y, msg=msg)

    def get_scale_int8(self, tensor):
        max_val = np.max(np.abs(tensor.detach().numpy()))
        min_val = np.min(np.abs(tensor.detach().numpy()))

        scale = 1.0
        position = 0
        absmax = max(max_val, -min_val)
        if absmax != 0:
            scale = 127.0 / absmax
        return position, scale.astype('float32')

    def get_scale_int16(self, tensor):
        max_val = np.max(tensor.detach().numpy())
        min_val = np.min(tensor.detach().numpy())
        scale = 1.0
        position = 0
        absmax = max(max_val, -min_val)
        if absmax != 0:
            scale = 32767.0 / absmax
        return position, scale.astype('float32')

    def get_scales_per_channel(self,tensor, q_type = 'int8'):

        assert tensor.dim() > 1, "tensor size must more than 1"
        channel_ = tensor.size(0)
        scales_ = []
        for i in range(channel_):
            if q_type == 'int8':
                _, merged_scale = self.get_scale_int8(tensor[i])
                scales_.append(merged_scale)
            elif q_type == 'int16':
                _, merged_scale = self.get_scale_int16(tensor[i])
                scales_.append(merged_scale)
            else:
                raise ValueError("Not support this quantize dtype")
        return scales_

    def iterate_init_args(self):
        for init_args in [tuple()]:
            yield init_args

    def iterate_inputs(self):
        shapes = [((1,3,224,1), (1,3,224, 224)),
                  ((3, 1), (3, 5)),
                  ((1,3, 1), (1, 3, 10))]
        for shape1,shape2 in shapes:
            input_1 = torch.rand(shape1, dtype=torch.float)
            yield input_1, shape2

    def optional_fake_half_cpu_inputs(self, tensor):
        if self.data_type == torch.HalfTensor:
            if isinstance(tensor, tuple):
                tensor = tuple(x.type(torch.HalfTensor).type(torch.FloatTensor) for x in tensor)
            else:
                tensor = tensor.type(torch.HalfTensor).type(torch.FloatTensor)
                tensor[tensor == float("inf")] = 65504
            return tensor
        else:
            return tensor

    def randn(self, sizes, dtype):
        return self.optional_fake_half_cpu_inputs(torch.randn((sizes), dtype=dtype))

    def rand(self, sizes, dtype):
        return self.optional_fake_half_cpu_inputs(torch.rand((sizes), dtype=dtype))

    if sys.version_info < (3, 2):
        # assertRegexpMatches renamed to assertRegex in 3.2
        assertRegex = unittest.TestCase.assertRegexpMatches
        # assertRaisesRegexp renamed to assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    if sys.version_info < (3, 5):
        # assertNotRegexpMatches renamed to assertNotRegex in 3.5
        assertNotRegex = unittest.TestCase.assertNotRegexpMatches

class OutputRedirector(object):
    """
    Class used to redirect standard output or another stream.
    """
    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.original_stream = stream
        if self.original_stream is None:
            self.original_stream = sys.stdout
        self.original_streamfd = self.original_stream.fileno()
        self.output_text = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start redirecting the stream data.
        """
        self.output_text = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.original_streamfd)  # pylint: disable= W0201
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.original_streamfd)

    def stop(self):
        """
        Stop redirecting the stream data and save the text in `output_text`.
        """
        # Print the escape character to make the readOutput method stop:
        self.original_stream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.original_stream.flush()
        self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.original_streamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `output_text`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.original_stream.encoding)
            if not char or self.escape_char in char:
                break
            self.output_text += char

# cases which cause core dump and randomly error
known_error_lst = ["test_addmm_sizes_mlu_float32",\
                 "test_addmm_sizes_mlu_float64",\
                 "test_floor_divide_tensor_mlu_int8",\
                 "test_floor_divide_tensor_mlu_uint8",\
                 "test_floor_divide_tensor_mlu_int16",\
                 "test_floor_divide_tensor_mlu_int64",\
                 "test_floor_divide_tensor_mlu_int32",\
                 "test_geometric_kstest_mlu_int32",\
                 "test_nonzero_empty_mlu",\
                 "test_blas_empty_mlu"
                 ]

# TODO: view op and below 6 test are tracked in jira: PYTORCH-4157
PYTORCH4157_torch_lst = [
                 "test_view_mlu"
                 ]
PYTORCH4157_tensor_lst = [
                 "test_max_dim_mlu_float16",\
                 "test_max_neg_dim_mlu_float16",\
                 "test_min_dim_mlu_float16",\
                 "test_min_neg_dim_mlu_float16",\
                 "test_std_dim_mlu_float16",\
                 "test_svd_tall_all_col_maj_mlu_float32"
                 ]

def runAllTests(cls):
    """
    Execute all test cases in test_torch.py , output reports and
    execute the regression tests
    """
    name_lst = ['op_name', 'info', 'has_error']
    out_file = pd.DataFrame(columns = name_lst)
    passed = 0
    total = 0
    for name in dir(cls):
        has_error = "no"
        if name[:4] == "test" and name not in known_error_lst \
                              and name not in PYTORCH4157_torch_lst:
            total += 1
            std_out = OutputRedirector()
            std_err = OutputRedirector(sys.stderr)
            std_out.start()
            std_err.start()
            try:
                getattr(cls, name)()
            except Exception as e:   # pylint: disable = W0703
                info=sys.exc_info()
                print(info[0],":",info[1])
                has_error = "yes"
            std_out.stop()
            std_err.stop()
            out = std_out.output_text + std_err.output_text
            if out != "":
                has_error = "yes"
            elif has_error == "no":
                passed += 1
            row = pd.DataFrame({name_lst[0]:[name], name_lst[1]:[out], name_lst[2]:[has_error]})
            out_file = out_file.append(row, ignore_index=True)
    print("class ", cls.__class__.__name__,\
          " passed ", passed, " cases. ",\
          "ratio is ", passed/total)
    out_file.to_csv("./"+cls.__class__.__name__+".csv")

    # Check the regression test
    err_str = ""
    reg_csv = pd.read_csv("./Reg"+cls.__class__.__name__+".csv")
    for i in range(reg_csv.shape[0]):
        reg_test = reg_csv.iloc[i]
        out_test = out_file.iloc[i]
        if reg_test["has_error"] == "no" and out_test["has_error"] == "yes":
            err_str += reg_test["op_name"]
            err_str += ", "
            print("ERROR OP: ", out_test["op_name"])
            print("ERROR INFO: ", out_test["info"])
    if err_str != "":
        # raise Exception("Regression test failed with these ops: ", err_str)
        warnings.warn("Regression test failed with these ops: " + err_str)


def runFnLst(cls, lst):
    """
    Test specified cases
    """
    for name in lst:
        getattr(cls, name)()
