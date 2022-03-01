#!/usr/bin/env python
#pylint: disable=C0301,W0613,W0611
from __future__ import print_function

import argparse
from datetime import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import logging

import torch
import torch._six
from torch.utils import cpp_extension
from common_utils import shell, print_to_stderr, gen_err_message

TESTS = [
    #'test_quantized_mods',
    #'test_quantize_generate',
    'test_notifier',
    'cnnl/op_test/test_all_operators0',
    'cnnl/op_test/test_all_operators1',
    'cnnl/op_test/test_all_operators2',
    'cnnl/op_test/test_all_operators3',
    'bang/op_test/test_all_operators',
    'cnnl/test_distributed',
    'cnnl/test_pin_memory',
    'cnnl/test_logging_cnnl',
    'cnnl/test_cnnl_op_exception',
    'cnnl/test_DEFAULT_MLU_DEVICE_NAME.py',
    'cnnl/test_op_methods_cnnl',
    'test_device',
    'test_mlu',
    #'test_dump',
    'test_caching_allocator',
    'test_queue',
    'cnnl_gtest',
    'common_gtest',
    'test_jit_exception',
    'cnnl/test_dumptool',
    'magicmind/op_test/test_all_operators0',
    'magicmind/op_test/test_all_operators1',
    'magicmind/op_test/test_all_operators2',
    'magicmind/op_test/test_all_operators3',
    'magicmind/test_lower_graph',
    'magicmind/test_segment_graph',
    'magicmind_gtest'
]

CNNL_BLACKLIST = [
    'cnnl/op_test/test_all_operators0',
    'cnnl/op_test/test_all_operators1',
    'cnnl/op_test/test_all_operators2',
    'cnnl/op_test/test_all_operators3',
    'cnnl/test_distributed',
    'cnnl/test_pin_memory',
    'cnnl/test_logging_cnnl',
    'cnnl/test_cnnl_op_exception',
    'cnnl_gtest',
]

MAGICMIND_BLACKLIST = [
    'magicmind/op_test/test_all_operators0',
    'magicmind/op_test/test_all_operators1',
    'magicmind/op_test/test_all_operators2',
    'magicmind/op_test/test_all_operators3',
    'magicmind/test_lower_graph',
    'magicmind/test_segment_graph',
    'magicmind_gtest'
]

def run_test(executable, test_module, test_directory, options, *extra_unittest_args):
    unittest_args = options.additional_unittest_args
    if options.verbose:
        unittest_args.append('--verbose')
    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test_module + '.py'] + unittest_args + list(extra_unittest_args)

    if options.coverage:
        command = ['coverage'] + ['run'] + argv
    else:
        command = executable + argv
    return shell(command, test_directory)

def get_backend_type(test_module):
    if 'magicmind' in test_module:
        return 'magicmind'
    elif 'cnnl' in test_module:
        return 'cnnl'
    else:
        raise RuntimeError("unsupported backend type, currently only support cnnl and magicmind.")

def test_operators0(executable, test_module, test_directory, options):
    backend_type = get_backend_type(test_module)
    operators_test_dir = os.path.join(test_directory, backend_type, 'op_test')
    return_code = shell([sys.executable, 'test_all_operators.py', '-n 0'], cwd=operators_test_dir)
    return return_code

def test_operators1(executable, test_module, test_directory, options):
    backend_type = get_backend_type(test_module)
    operators_test_dir = os.path.join(test_directory, backend_type, 'op_test')
    return_code = shell([sys.executable, 'test_all_operators.py', '-n 1'], cwd=operators_test_dir)
    return return_code

def test_operators2(executable, test_module, test_directory, options):
    backend_type = get_backend_type(test_module)
    operators_test_dir = os.path.join(test_directory, backend_type, 'op_test')
    return_code = shell([sys.executable, 'test_all_operators.py', '-n 2'], cwd=operators_test_dir)
    return return_code

def test_operators3(executable, test_module, test_directory, options):
    backend_type = get_backend_type(test_module)
    operators_test_dir = os.path.join(test_directory, backend_type, 'op_test')
    return_code = shell([sys.executable, 'test_all_operators.py', '-n 3'], cwd=operators_test_dir)
    return return_code

def test_executable_file(executable, test_module, test_directory, options):
    gtest_dir = os.path.join(test_directory,'../build/bin')
    if 'cnnl_gtest' in test_module:
        gtest_dir = os.path.join(gtest_dir,'cnnl')
    elif 'magicmind_gtest' in test_module:
        gtest_dir = os.path.join(gtest_dir,'magicmind')
    else:
        gtest_dir = os.path.join(gtest_dir,'common')

    total_error_info = []
    if os.path.exists(gtest_dir):
        commands = (os.path.join(gtest_dir, filename) for filename in os.listdir(gtest_dir))
        for command in commands:
            return_code = shell([command,], test_directory)
            gen_err_message(return_code, command, total_error_info)

    # Print total error message
    print("*********** Gtest : Error Message Summaries **************")
    for err_message in total_error_info:
        logging.error("\033[31;1m {}\033[0m .".format(err_message))
    print("**********************************************************")

    return 1 if total_error_info else 0

CUSTOM_HANDLERS = {
    'cnnl/op_test/test_all_operators0': test_operators0,
    'cnnl/op_test/test_all_operators1': test_operators1,
    'cnnl/op_test/test_all_operators2': test_operators2,
    'cnnl/op_test/test_all_operators3': test_operators3,
    'magicmind/op_test/test_all_operators0': test_operators0,
    'magicmind/op_test/test_all_operators1': test_operators1,
    'magicmind/op_test/test_all_operators2': test_operators2,
    'magicmind/op_test/test_all_operators3': test_operators3,
    'cnnl_gtest': test_executable_file,
    'common_gtest': test_executable_file,
    'magicmind_gtest': test_executable_file,
}

def parse_test_module(test):
    return test.split('.')[0]

class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super(TestChoices, self).__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the PyTorch unit test suite',
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)))
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print verbose information and test-by-test results')
    parser.add_argument(
        '-i',
        '--include',
        nargs='+',
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar='TESTS',
        help='select a set of tests to include (defaults to ALL tests).'
             ' tests can be specified with module name, module.TestClass'
             ' or module.TestClass.test_method')
    parser.add_argument(
        '-x',
        '--exclude',
        nargs='+',
        choices=TESTS,
        metavar='TESTS',
        default=[],
        help='select a set of tests to exclude')
    parser.add_argument(
        '-f',
        '--first',
        choices=TESTS,
        metavar='TESTS',
        help='select the test to start from (excludes previous tests)')
    parser.add_argument(
        '-l',
        '--last',
        choices=TESTS,
        metavar='TESTS',
        help='select the last test to run (excludes following tests)')
    parser.add_argument(
        '--bring-to-front',
        nargs='+',
        choices=TestChoices(TESTS),
        default=[],
        metavar='TESTS',
        help='select a set of tests to run first. This can be used in situations'
             ' where you want to run all tests, but care more about some set, '
             'e.g. after making a change to a specific component')
    parser.add_argument(
        '--ignore_cnnl_blacklist',
        action='store_true',
        help='always ignore blacklisted train tests')
    parser.add_argument(
        '--ignore_magicmind_blacklist',
        action='store_true',
        help='always ignore blacklisted magicmind eval tests')
    parser.add_argument(
        'additional_unittest_args',
        nargs='*',
        help='additional arguments passed through to unittest, e.g., '
             'python run_test.py -i sparse -- TestSparse.test_factory_size_check')
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='test python code coverage')
    return parser.parse_args()


def get_executable_command(options):
    executable = [sys.executable]
    return executable

def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.

    This function is used to determine the indices when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False, result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True, result should be **4**.

    Arguments:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurance of the given test
    """
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            found_idx = idx
            if not find_last_index:
                break
        idx += 1
    return found_idx


def exclude_tests(exclude_list, selected_tests, exclude_message=None):
    tests_copy = selected_tests[:]
    for exclude_test in exclude_list:
        for test in tests_copy:
            if test.startswith(exclude_test):
                if exclude_message is not None:
                    print_to_stderr('Excluding {} {}'.format(test, exclude_message))
                selected_tests.remove(test)
    return selected_tests


def get_selected_tests(options):
    selected_tests = options.include

    if options.bring_to_front:
        to_front = set(options.bring_to_front)
        selected_tests = options.bring_to_front + list(filter(lambda name: name not in to_front,
                                                              selected_tests))

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[:last_index + 1]

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if not options.ignore_cnnl_blacklist:
        selected_tests = exclude_tests(CNNL_BLACKLIST, selected_tests)

    if not options.ignore_magicmind_blacklist:
        selected_tests = exclude_tests(MAGICMIND_BLACKLIST, selected_tests)

    return selected_tests

def main():
    options = parse_args()
    executable = get_executable_command(options)  # this is a list
    print_to_stderr('Test executor: {}'.format(executable))
    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)
    total_error_info = []

    if options.verbose:
        print_to_stderr('Selected tests: {}'.format(', '.join(selected_tests)))

    for test in selected_tests:
        test_module = parse_test_module(test)

        # Printing the date here can help diagnose which tests are slow
        print_to_stderr('Running {} ... [{}]'.format(test, datetime.now()))
        handler = CUSTOM_HANDLERS.get(test, run_test)
        return_code = handler(executable, test_module, test_directory, options)
        assert isinstance(return_code, int) and not isinstance(
            return_code, bool), 'Return code should be an integer'
        gen_err_message(return_code, test, total_error_info)
    # Print total error message
    print("***************** run_test.py : Error Message Summaries **********************")
    for err_message in total_error_info:
        logging.error("\033[31;1m {}\033[0m .".format(err_message))
    print("*******************************************************************************")

    if total_error_info:
        raise RuntimeError("run_test case Failed")


if __name__ == '__main__':
    main()
