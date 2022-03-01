from __future__ import print_function

import sys
import os
import logging
from fnmatch import fnmatch  # pylint: disable=C0413
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import shell, gen_err_message  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


def get_executable_command():
    executable = [sys.executable]
    return executable

def get_test_files(path):
    pyfiles = [name for name in os.listdir(path) if fnmatch(name, '*.py')]
    this_file = __file__.split('/')[-1]
    if this_file in pyfiles:
        pyfiles.remove(this_file)
    return pyfiles

def run_test(executable, test_files, test_directory):
    total_error_info = []
    commands = (executable + [argv] for argv in test_files)
    for command in commands:
        return_code = shell(command, test_directory)
        gen_err_message(return_code, command[-1], total_error_info)

    # Print total error message
    print("*********** BANG test_all_operators.py : Error Message Summaries **************")
    for err_message in total_error_info:
        logging.error("\033[31;1m {}\033[0m .".format(err_message))
    print("*******************************************************************************")

    if total_error_info:
        raise RuntimeError("BANG test_all_operators case Failed")

if __name__ == '__main__':
    if not os.getenv('MLU_VISIBLE_DEVICES'):
        os.environ['MLU_VISIBLE_DEVICES']="0"
    executable_ = get_executable_command()
    pyfiles_ = get_test_files(cur_dir)
    run_test(executable_, pyfiles_, cur_dir)
