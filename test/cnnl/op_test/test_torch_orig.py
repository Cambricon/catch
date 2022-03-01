from __future__ import print_function

import sys
import logging
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from common_utils import OutputRedirector, runFnLst, runAllTests
import torch_mlu.core.mlu_model as ct
os.environ['ENABLE_CNNL_TRYCATCH'] = 'OFF' # pylint: disable=C0413
pytorch_dir = os.getenv("PYTORCH_HOME")
if pytorch_dir is None:
    raise Exception("Pytorch directory does not set!!")
sys.path.append(pytorch_dir + "/test/")
from test_torch import TestTorchDeviceTypeMLU
logging.basicConfig(level=logging.DEBUG)

torch_device_lst = ["test_diagonal_mlu"] # Run specified cases

'''
MLUTestTorchDeviceType tests the original pytorch test: test_torch.py.
Add the test name into the test_lst for running specified test cases.
TO BE NOTICED. the original decorator: onlyCPU and onlyGPU must be deprecated
through patches, otherwise the new decorator runMLU can not work!
'''

if __name__ == '__main__':
    ct._jit_override_can_fuse_on_mlu(False)
    test_1 = TestTorchDeviceTypeMLU()
    runFnLst(test_1, torch_device_lst)
    print("TestTorchDeviceType tests finished!!!")

    # run all tests and output tests report
    runAllTests(test_1)
    ct._jit_override_can_fuse_on_mlu(True)
