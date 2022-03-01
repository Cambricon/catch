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
from test_torch import TestTensorDeviceOpsMLU
logging.basicConfig(level=logging.DEBUG)

tensor_device_lst = ["test_abs_mlu_float32"]

'''
MLUTestTensorDeviceType tests the original pytorch test: test_torch.py.
Add the test name into the test_lst for running specified test cases.
TO BE NOTICED. the original decorator: onlyCPU and onlyGPU must be deprecated
through patches, otherwise the new decorator runMLU can not work!
'''

if __name__ == '__main__':
    test_2 = TestTensorDeviceOpsMLU()
    runFnLst(test_2, tensor_device_lst)
    print("TestTensorDeviceOps tests finished!!!")

    # run all tests and output tests report
    runAllTests(test_2)
