#!/usr/bin/env python
from __future__ import print_function
import os
import re
import sys
import stat
import glob
import shutil
import json
import subprocess
import torch

import distutils.ccompiler  # pylint: disable=C0411
import distutils.command.clean  # pylint: disable=C0411
import setuptools.command.install  # pylint: disable=C0411

from setuptools import setup, find_packages, distutils  # pylint: disable=C0411
from torch.utils.cpp_extension import BuildExtension, CppExtension  # pylint: disable=C0411


################################################################################
# Parameters parsed from environment
################################################################################
RUN_BUILD_CORE_LIBS = True
RUN_AUTO_GEN_CATCH_CODE = True
RUN_BUILD_ASAN_CHECK = False
RUN_BUILD_WARNING_CHECK = True

RUN_BUILD_USE_PYTHON = bool((os.getenv('USE_PYTHON') is None)
        or (os.getenv('USE_PYTHON').upper()
            not in ['OFF', '0', 'NO', 'FALSE', 'N']))
RUN_BUILD_USE_BANG = bool((os.getenv('USE_BANG') is None)
        or (os.getenv('USE_BANG').upper()
            not in ['OFF', '0', 'NO', 'FALSE', 'N']))

for i, arg in enumerate(sys.argv):
    if arg == 'clean':
        RUN_BUILD_CORE_LIBS = False
        RUN_AUTO_GEN_CATCH_CODE = False

# Get the current path, core library paths and neuware path
cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch_mlu", "csrc", "lib")

# NEUWARE_HOME env must be set before compiling
if not os.getenv('NEUWARE_HOME'):
    print("[Error] NEUWARE_HOME Environment Variable has not been set,",
           "Please firstly get and install the Cambricon Neuware Package,",
           "then use NEUWARE_HOME to point it!")
    sys.exit()

# Get Pytorch Dir
base_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_source_path = os.getenv('PYTORCH_HOME', os.path.dirname(base_dir))

# lib/pythonx.x/site-packages
rel_site_packages = distutils.sysconfig.get_python_lib(prefix='')
# full absolute path to the dir above
full_site_packages = distutils.sysconfig.get_python_lib()

# Define the compile and link options
extra_link_args = []
extra_compile_args = []

# Check env flag
def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

def _check_env_off_flag(name, default=''):
    return os.getenv(name, default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']

# make relative rpath
def make_relative_rpath(path):
    return '-Wl,-rpath,$ORIGIN/' + path

# Generate parts of header/source files in Catch automatically
def gen_catch_code():
    if RUN_BUILD_USE_BANG:
        os.system('python torch_mlu/tools/gen.py --use_bang')
    else:
        os.system('python torch_mlu/tools/gen.py')

# Calls build_catch_lib.sh with the corrent env variables
def build_libs():
    build_libs_cmd = ['bash', os.path.join('..', 'script', 'build_catch_lib.sh')]
    my_env = os.environ.copy()

    my_env["PYTORCH_PYTHON"] = sys.executable
    my_env['CMAKE_INSTALL'] = 'make install'

    cmake_prefix_path = full_site_packages
    if "CMAKE_PREFIX_PATH" in my_env:
        cmake_prefix_path = my_env["CMAKE_PREFIX_PATH"] + ";" + cmake_prefix_path
    my_env["CMAKE_PREFIX_PATH"] = cmake_prefix_path

    my_env["PYTORCH_SOURCE_PATH"] = pytorch_source_path

    # Keep the same compile and link args between setup.py and build_catch_lib.sh
    my_env["EXTRA_COMPILE_ARGS"] = ' '.join(extra_compile_args)
    my_env["EXTRA_LINK_ARGS"] = ' '.join(extra_link_args)

    # set up the gtest compile runtime environment.
    my_env['BUILD_TEST'] = 'ON' if _check_env_flag('BUILD_TEST') else 'OFF'
    my_env["USE_PYTHON"] = 'OFF' if _check_env_off_flag('USE_PYTHON') else 'ON'
    my_env["USE_BANG"] = 'OFF' if _check_env_off_flag('USE_BANG') else 'ON'
    my_env["USE_CNCL"] = 'OFF' if _check_env_off_flag('USE_CNCL') else 'ON'
    my_env["USE_MAGICMIND"] = 'OFF' if _check_env_off_flag('USE_MAGICMIND') else 'ON'
    my_env["USE_PROFILE"] = 'OFF' if _check_env_off_flag('USE_PROFILE') else 'ON'

    # ABI version
    abi_version=(int)(torch.compiled_with_cxx11_abi())
    my_env["GLIBCXX_USE_CXX11_ABI"] = str(abi_version)

    try:
        os.mkdir('build')
    except OSError:
        pass

    kwargs = {'cwd': 'build'}

    if subprocess.call(build_libs_cmd, env=my_env, **kwargs) != 0:
        print("Failed to run '{}'".format(' '.join(build_libs_cmd))) # pylint: disable=C0209
        sys.exit(1)

class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)

class Build(BuildExtension):
    def run(self):
    # Run the original BuildExtension first. We need this before building
    # the tests.
        BuildExtension.run(self)

class Clean(distutils.command.clean.clean):
    def run(self):
        try:
            with open('.gitignore', 'r') as f: # pylint: disable=W1514
                ignores = f.read()
                pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
                for wildcard in filter(None, ignores.split('\n')):
                    match = pat.match(wildcard)
                    if match:
                        if match.group(1):
                            # Marker is found and stop reading .gitignore.
                            break
                        # Ignore lines which begin with '#'.
                    else:
                        for filename in glob.glob(wildcard):
                            try:
                                os.remove(filename)
                            except OSError:
                                shutil.rmtree(filename, ignore_errors=True)
        except OSError:
            shutil.rmtree('build', ignore_errors=True)
        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

# Configuration for Build the Project.
main_libraries = ['catch_python']
include_dirs = []
library_dirs = []

# Fetch the sources to be built.
torch_mlu_sources = (
    glob.glob('torch_mlu/csrc/stub.cpp')
)

# include head files
include_dirs += [
    base_dir,
    os.path.join(pytorch_source_path, 'aten', 'src'),
    os.path.join(pytorch_source_path, 'torch', 'csrc'),
    os.path.join(pytorch_source_path, 'torch', 'csrc', 'api', 'include'),
    os.path.join(pytorch_source_path, 'torch', 'lib', 'tmp_install', 'include'),
]

#include lib files
library_dirs.append(lib_path)

extra_compile_args += [
    '-std=c++14',
    '-pthread',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type',
    '-Werror',
]

DEBUG = _check_env_flag('DEBUG')
if RUN_BUILD_ASAN_CHECK:
    # To get a reasonable performace add -O1 or higher.
    # run executable with LD_PRELOAD=path/to/asan/runtime/lib
    extra_compile_args += ['-O1', '-g', '-DDEBUG',
        '-fsanitize=address', '-fno-omit-frame-pointer']
elif DEBUG:
    extra_compile_args += ['-Og', '-g', '-DDEBUG']
else:
    extra_compile_args += ['-O3']

TEST_COVERAGE = _check_env_flag('TEST_COVERAGE')
if TEST_COVERAGE:
    extra_compile_args += ['-fprofile-arcs', '-ftest-coverage']
    extra_link_args += ['-fprofile-arcs', '-ftest-coverage']
    #to test coverage, these args are necessary

config_dir = os.path.join(cwd, 'cmake')
modules_dir = os.path.join(cwd, 'torch_mlu/share/cmake/TorchMLU/modules')
if not os.path.exists(modules_dir):
    os.makedirs(modules_dir)
shutil.copy(base_dir + '/cmake/modules/FindCNNL.cmake', modules_dir)
shutil.copy(base_dir + '/cmake/modules/FindCNRT.cmake', modules_dir)
shutil.copy(base_dir + '/cmake/modules/FindCNDRV.cmake', modules_dir)
shutil.copy(base_dir + '/cmake/modules/FindCNLIGHT.cmake', modules_dir)

if _check_env_flag('USE_MAGICMIND'):
    shutil.copy(base_dir + '/cmake/modules/FindMAGICMIND.cmake', modules_dir)

# Generate parts of Catch code
if RUN_AUTO_GEN_CATCH_CODE:
    gen_catch_code()

# Build Catch Core Libs
if RUN_BUILD_CORE_LIBS:
    build_libs()

if not RUN_BUILD_USE_PYTHON:
    sys.exit(0)

json_file = os.path.join(cwd, 'script/release', 'build.property')
torch_mlu_version = 'unknown'
if os.path.isfile(json_file):
    with open(json_file, 'r') as f: # pylint: disable=W1514
        json_dict = json.load(f)
        torch_mlu_version = json_dict['version'].strip()

# Setup
setup(
    name='torch_mlu',
    version=torch_mlu_version,
    description='MLU bridge for PyTorch',
    # Exclude the build files.
    packages=find_packages(exclude=['build']),
    ext_modules=[
        CppExtension(
            'torch_mlu._MLUC',
            libraries=main_libraries,
            sources=torch_mlu_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + [make_relative_rpath('csrc/lib')]),
    ],
    package_data={
        'torch_mlu': [
            'csrc/lib/*.so*',
            'share/cmake/TorchMLU/TorchMLUConfig.cmake',
            'share/cmake/TorchMLU/modules/*.cmake',
            'csrc/aten/cnnl/*.h',
            'csrc/aten/core/*.h',
            'csrc/aten/device/*.h',
            'csrc/aten/operators/op_proxy.h',
            'csrc/aten/util/*.h',
            'csrc/aten/util/stash/*.h',
            'csrc/api/include/torch_mlu/*.h',
            'csrc/api/include/torch_mlu/nn/*.h',
            'csrc/api/include/torch_mlu/nn/modules/*.h',
            'csrc/api/include/torch_mlu/nn/functional/*.h',
            'csrc/api/include/torch_mlu/operators/*.h',
            'csrc/api/include/torch_mlu/quantized/*.h',
            'csrc/aten/operators/cnnl_ops.h',
            'csrc/aten/operators/op_methods.h',
            "csrc/aten/operators/cnnl/*.h",
            "csrc/aten/operators/cnnl/internal/*.h",
            "csrc/aten/generated/autograd/functions.h",
            "csrc/jit/interface.h"
        ],
    },
    cmdclass={
        'build_ext': Build,
        'clean': Clean,
        'install': install,
    })
