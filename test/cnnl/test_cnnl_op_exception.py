import unittest
import sys
import os
import subprocess
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import testinfo, TestCase # pylint: disable=C0413,C0411

class TestCnnlOpException(TestCase):

    @testinfo()
    #@unittest.skip("not test")
    def test_unsupported_shapeinfo(self):
        print("****Test CNNL op failure: unsupported shape info of input****")
        execute_list = [
            'import torch' + '\n',
            'import torch_mlu' + '\n',
            'import torch_mlu.core.mlu_model as ct' + '\n',
            'x1 = torch.rand(64, 2048)' + '\n',
            'x2 = torch.rand(1048, 1000)' + '\n',
            'out = torch.mm(x1.to(ct.mlu_device()), x2.to(ct.mlu_device()))'
        ]

        # enable the log info of CNLOG(DBG).
        os.environ['TORCH_MIN_CNLOG_LEVEL'] = '-1'
        with open(os.path.join(cur_dir, 'execute.py'), 'w+') as f:
            f.writelines(execute_list)

        cmd = 'python ' + os.path.join(cur_dir, 'execute.py') + \
              ' 2> ' + os.path.join(cur_dir, 'console.txt')

        check_info_flag = False
        # run mlu forward
        subprocess.call(cmd, shell=True)
        with open(os.path.join(cur_dir, 'console.txt'), 'r') as f:
            for line in f:
                # check the shape info in the output log.
                if 'm1: ' in line and '[64, 2048]' in line and \
                   '[1048, 1000]' in line and 'size mismatch' in line:
                    check_info_flag = True

        os.remove(os.path.join(cur_dir, 'execute.py'))
        os.remove(os.path.join(cur_dir, 'console.txt'))
        self.assertTrue(check_info_flag,
                        "size mismatch, m1: [64, 2048], m2: [1048, 1000]")

    @testinfo()
    #@unittest.skip("not test")
    def test_inconsistant_datatype(self):
        print("****Test CNNL op failure: inconsistant input datatype****")
        execute_list = [
            'import torch' + '\n',
            'import torch_mlu' + '\n',
            'import torch_mlu.core.mlu_model as ct' + '\n',
            'in_shape1 = (1, 2, 3)' + '\n',
            'in_shape2 = (1, 77, 3)' + '\n',
            'input1 = torch.ones(in_shape1, dtype=torch.float)' + '\n',
            'input2 = torch.ones(in_shape2, dtype=torch.half)' + '\n',
            'inputs_mlu = [input1.to(ct.mlu_device()), input2.to(ct.mlu_device())]' + '\n',
            'out = torch.cat(inputs_mlu, dim=1)'
        ]

        # enable the log info of CNLOG(DBG).
        os.environ['TORCH_MIN_CNLOG_LEVEL'] = '-1'
        with open(os.path.join(cur_dir, 'execute.py'), 'w+') as f:
            f.writelines(execute_list)

        cmd = 'python ' + os.path.join(cur_dir, 'execute.py') + \
              ' 2> ' + os.path.join(cur_dir, 'console.txt')

        check_info_flag = False
        check_cnnl_flag = False
        # run mlu forward
        subprocess.call(cmd, shell=True)
        with open(os.path.join(cur_dir, 'console.txt'), 'r') as f:
            for line in f:
                # check the shape info in the output log.
                if '[cat]' in line and 'shape: [1, 2, 3]' in line and \
                   'shape: [1, 77, 3]' in line and 'device: mlu' in line and \
                   'dtype: Half' in line:
                    check_info_flag = True
                elif 'CNNL_STATUS_BAD_PARAM' in line:
                    check_cnnl_flag = True

        os.remove(os.path.join(cur_dir, 'execute.py'))
        os.remove(os.path.join(cur_dir, 'console.txt'))
        self.assertTrue(check_info_flag and check_cnnl_flag,
                        "Catch doesn't give the detailed information when " +
                        "CNNL_STATUS_BAD_PARAM error occurs, Please try to resolve " +
                        "the problem by yourself or report a bug to Cambricon Pytorch Team.")

if __name__ == '__main__':
    unittest.main()
