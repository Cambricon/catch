'''
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch_mlu
import warnings
import os

def dump_cnnl_gencase(enable=True, level='L1'):
    r'''
    Dump imformations about CNNL Kernels.

    Arguments:
        enable (bool): Turn on/off the cnnl gencase API.
        level (str): Three mode that cnnl API provide,
            'L1' : dump shapes, dtype, layout and args of CNNL kernels to files.
            'L2' : dump shapes, dtype, layout, args and input real data of CNNL Kernels to files,
                   will take a long time.
            'L3' : print shapes, dtype layout and args of CNNL Kernels to screen.

    Return :
         The cnnl gencase API state(on/off).
    '''
    cnnl_gencase_env = os.environ.get('CNNL_GEN_CASE')
    if cnnl_gencase_env is not None:
        warnings.warn("Please unset enviroment variable 'CNNL_GEN_CASE',"
                      "or the dump_cnnl_gencase method will not work.")
        return False

    _level = ['L1', 'L2', 'L3']
    if not enable:
        torch_mlu._MLUC._dump_cnnl_gencase(0)
        return False
    if level not in _level:
        msg = ("dump_cnnl_gencase only support level"
               "'L0' 'L1' and 'L2', but get level {}.".format(level))
        raise ValueError(msg)

    if level == _level[0]:
        torch_mlu._MLUC._dump_cnnl_gencase(1)
    elif level == _level[1]:
        torch_mlu._MLUC._dump_cnnl_gencase(2)
    else:
        torch_mlu._MLUC._dump_cnnl_gencase(3)
    return True

def start_dump(dump_dir="./dump", enable=True, use_cpu=False, level=0):
    torch_mlu._MLUC._dump_start(dump_dir, enable, use_cpu, level)

def end_dump():
    torch_mlu._MLUC._dump_finish()

class Dumper():
    def __init__(self, dump_dir="./dump",enable=True, use_cpu=False, level=0):
        self.dump_dir = dump_dir
        self.enable = enable
        self.use_cpu = use_cpu
        self.level = level

    def __enter__(self):
        start_dump(self.dump_dir, self.enable, self.use_cpu, self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_dump()
        return False
