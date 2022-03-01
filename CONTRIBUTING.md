# Contributing to PyTorch/CATCH

If you are interested in contributing to CATCH, your contributions will fall
into two categories:

1. You want to propose a new feature and implement it.
    - Post about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue here: https://github.com/Cambricon/catch/issues
    - Pick an issue and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/Cambricon/catch

# Developing CATCH

When you are installing from source, you will need a C++14 compiler. Also, we highly recommend installing an Anaconda or Virtualenv environment.
You will get controlled dependency versions regardless of your Linux distro.

To develop CATCH on your machine, here are some tips:

* Get and install the Cambricon Neuware Package

CATCH depends on the Cambricon Neuware Package to compile, so you should get and install the
Cambricon Neuware Package before compiling. You can send the email to service@cambricon.com or
contact with the Cambricon engineer to get the package. For more information, you can refer the
Cambricon PyTorch User's Guide.

* Clone a copy of PyTorch from source and checkout the required branch:

```bash
git clone https://github.com/pytorch/pytorch.git -b v1.6.0
```

* Clone a copy of CATCH from source:

```bash
git clone https://github.com/Cambricon/catch.git
```

* Clone a copy of Vision from source and checkout the required branch:

```bash
git clone https://github.com/pytorch/vision.git -b v0.7.0
```

## Build From Source

* Apply PyTorch patches:

```bash
export PYTORCH_HOME=your pytorch repo root path
bash catch/script/apply_patches_to_pytorch.sh
```

* Build PyTorch source:

```bash
cd pytorch
pip install -r requirements.txt
python setup.py install
```

* Build CATCH source:

```bash
cd catch
pip install -r requirements.txt
python setup.py install
```

* Build Vision source:

```bash
cd vision
pip install -r requirements.txt
python setup.py install
```

## Building With Script

* Build and install `torch`, `torch_mlu`, `torchvision`

```bash
export PYTORCH_HOME=your pytorch repo root path
export VISION_HOME=your vision repo root path
export NEUWARE_HOME=your neuware package root path

bash catch/script/build_catch.sh 0 0
```

# Unit testing

CATCH's testing is located under `test/`. Run the entire test suite with

```bash
python test/run_test.py
```

or run individual test files, like `python test/test_caching_allocator.py`, for individual test suites.
