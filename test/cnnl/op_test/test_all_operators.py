from __future__ import print_function

import sys
import os
import logging
import argparse
from fnmatch import fnmatch  # pylint: disable=C0413
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import shell, gen_err_message  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)

def printfilesinfo(pyfiles):
    print("total pyfiles")
    print("total" + str(len(pyfiles)))
    print(pyfiles)

def get_executable_command():
    executable = [sys.executable]
    return executable

def get_file_starting_character(cur_dir, initial):
    pyfiles = []
    for c in initial:
        pattern = 'test_' + c + '*.py'
        pyfiles += get_test_files(cur_dir, pattern)
    return pyfiles

def get_test_files(path, pattern):
    pyfiles = [name for name in os.listdir(path) if fnmatch(name, pattern)]
    this_file = __file__.rsplit('/', maxsplit=1)[-1]
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
    print("*********** CNNL test_all_operators.py : Error Message Summaries **************")
    for err_message in total_error_info:
        logging.error("\033[31;1m {}\033[0m .".format(err_message))
    print("*******************************************************************************")

    if total_error_info:
        raise RuntimeError("cnnl test_all_operators case Failed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=4, help = 'test op initial', type = int)
    args = parser.parse_args()
    if not os.getenv('MLU_VISIBLE_DEVICES'):
        os.environ['MLU_VISIBLE_DEVICES']="0"
    executable_ = get_executable_command()
    if args.n == 0:
        pyfiles_ = get_file_starting_character(cur_dir, 'acd')
        run_test(executable_, pyfiles_, cur_dir)
    elif args.n == 1:
        pyfiles_ = get_file_starting_character(cur_dir, 'efghijkln')
        run_test(executable_, pyfiles_, cur_dir)
    elif args.n == 2:
        pyfiles_ = get_file_starting_character(cur_dir, 'mopq')
        run_test(executable_, pyfiles_, cur_dir)
    elif args.n == 3:
        pyfiles_ = get_file_starting_character(cur_dir, 'brstuvwxyz')
        run_test(executable_, pyfiles_, cur_dir)
    elif args.n == 4:
        pyfiles_ = get_test_files(cur_dir, 'test_*.py')
        run_test(executable_, pyfiles_, cur_dir)
    else:
        print("unsupport parameter")
        sys.exit()
