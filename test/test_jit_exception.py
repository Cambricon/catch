import unittest
import os
os.environ["DISABLE_MLU_FUSION"] = '1'
import logging
import torch_mlu
import torch_mlu.core.mlu_model as ct

import sys
#sys.path.append("../")
from common_utils import testinfo, TestCase
import torch

class TestIfElseModel(torch.nn.Module):
    def __init__(self):
        super(TestIfElseModel, self).__init__()

    def forward(self, x, y, z):
        if x.size(0) > y.size(1):
            return x
        elif x.size(0) < y.size(1):
            return x,y
        else:
            return x,y,z

class TestForLoopModel(torch.nn.Module):
    def __init__(self):
        super(TestForLoopModel, self).__init__()

    def forward(self, x):
        for i in range(x.size(0)):
            x = x + 1
        return x

class TestReturnType(torch.nn.Module):
    def __init__(self):
        super(TestReturnType, self).__init__()

    def forward(self, x, y):
        out = dict()
        z = x+y
        out["1"] = x
        out["2"] = y
        out["3"] = z
        return out

class TestInputType(torch.nn.Module):
    def __init__(self):
        super(TestInputType, self).__init__()

    def forward(self, x):
        for k in x.keys():
            tmp = x[k]
        return tmp

class TestJitException(TestCase):
    @testinfo()
    def test_if_elif_else(self):
        print("********Test if-else statements********")
        ex1 = torch.rand(1,2)
        ex2 = torch.rand(1,2)
        ex3 = torch.rand(1,2)
        model = TestIfElseModel()
        traced_model = torch.jit.trace(model, (ex1, ex2, ex3), check_trace=False)

        in1 = torch.rand(2,1)
        in2 = torch.rand(2,1)
        in3 = torch.rand(2,1)
        output1 = traced_model(ex1,ex2,ex3)
        output2 = traced_model(in1,in2,in3)
        if type(output1) == type(output2):
            print("")
            logging.error("Torch.jit.trace cannot perfectly support if-else statements.")
            logging.error("In this test case:")
            logging.error("  When tracing the model,")
            logging.error("  according to the shapes of example input tensors,")
            logging.error("  it goes through the elif (x.size(0) < y.size(1)): banrch.")
            print("")
            logging.error("  When doing inference on traced_model,")
            logging.error("  the forward function has been fixed to")
            logging.error("  run on elif (x.size(0) < y.size(1))")
            logging.error("  branch no matter what the real inputs are.")
            print("")
            logging.error("Please check are there any if-else statements in your pytorch model.")
            print("")

    @testinfo()
    def test_for_loop(self):
        print("********Test for loop********")
        ex = torch.zeros(4)
        model = TestForLoopModel()
        traced_model = torch.jit.trace(model, ex, check_trace=True)

        x = torch.zeros(5)
        output1 = traced_model(x)
        output2 = traced_model(ex)

        if output1.max() == output2.max():
            print("")
            logging.error("Torch.jit.trace cannot perfectly support for-loop when the number")
            logging.error("of cycle is not fixed(e.g. cycle number depends on the input(s).")
            logging.error("In this test case:")
            logging.error("  The cycle number in traced function depends on the shape of input tensor.")
            logging.error("  When tracing the model, according the the shape of input tensor (ex.size(0) = 4),")
            logging.error("  the range of for loop in traced function has been fixed to 0~4.")
            logging.error("  When doing inference on traced_model, the range of foor loop is always 0~4 no matter what")
            logging.error("  the shape of real input is.")
            logging.error("  This raises the logic problem in futher use.")
            logging.error("Plase check are there any indifinite cycles of for loop in your pytorch model.")
            print("")


    @testinfo()
    def test_wrong_return_type(self):
        print("********Test wrong return type********")
        ex1 = torch.rand(1,2)
        ex2 = torch.rand(1,2)
        model = TestReturnType()
        try:
            traced_model = torch.jit.trace(model, (ex1, ex2), check_trace=True)
        except:
            print("")
            logging.error("Torch.jit.trace can only support tensor/tuples of tensors for return type.")
            logging.error("In this test case:")
            logging.error("  The return type of traced function is <class 'dict'>,")
            logging.error("  which is not supported by torch.jit.trace.")
            logging.error("  Other types such as, <class 'int'>, <class 'list'> are also")
            logging.error("  not supportted by traced function.")
            logging.error("Please check is there any unsupported return types in your pytorch model.")
            print("")

    @testinfo()
    def test_wrong_input_type(self):
        print("********Test wrong input type********")
        input1 = dict()
        input1["1"] = 1
        input1["2"] = 2
        input1["3"] = 3
        model = TestInputType()
        try:
            traced_mode = torch.jit.trace(model, input1, check_trace=True)
        except:
            print("")
            logging.error("Torch.jit.trace can only support tensors and (possibly nested) lists, dicts,")
            logging.error("and tuples of tensors for input type.")
            logging.error("In this test case:")
            logging.error("  The input type of traced functions is Tuple[Dict[str, int]],")
            logging.error("  which is not supported by torch.jit.trace.")
            logging.error("Please check is there any unsupported input type in your pytorch model.")
            print("")

if __name__ == '__main__':
    unittest.main()
