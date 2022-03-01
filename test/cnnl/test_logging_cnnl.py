from __future__ import print_function
import os
import sys

import unittest
import logging
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413
logging.basicConfig(level=logging.DEBUG)

class TestLoggingCNNL(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_cnnl_logging(self):
        data_path = os.path.join(cur_dir, '../data/cnlog/')
        cmd = 'python ' + os.path.join(data_path, 'cnlog_cnnl.py') + \
          ' 2>' + os.path.join(data_path, 'cnlog_cnnl.txt')
        os.system(cmd)
        with open(os.path.join(data_path, 'cnlog_cnnl.txt'), 'r') as f1:
            temp = f1.readlines()
        with open(os.path.join(data_path, 'cnlog_cnnl.err'), 'r') as f2:
            msg2 = f2.readlines()
        msg1 = []
        for line in temp:
            if line.startswith('[DEBUG]'):
                msg1.append(line)

        for i, line in enumerate(msg2):
            self.assertNotEqual(msg1[i].find(line), -1)

        os.remove(os.path.join(data_path, 'cnlog_cnnl.txt'))

if __name__ == '__main__':
    unittest.main()
