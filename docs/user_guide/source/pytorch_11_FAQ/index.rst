FAQ
==========================

编译安装
----------------------------

Q：安装依赖或源码编译过程中出现 ``unrecognized command line option xxx`` 类似报错，例如：

   ::
   
     "unrecognized command line option ‘-fno-plt’"
     "unrecognized command line option ‘c++11’"
     "unrecognized command line option ‘c++14’"

A：由于环境中g++版本较低导致无法识别指定的编译参数，建议将g++版本升级为大于等于5.4.0。

Q：执行 ``python setup.py install`` 安装Torchvision出现编译报错：

   ::
   
      error: cannot convert 'std::nullptr_t' to 'Py_ssize_t {aka long int}' in initialization.

A：安装Torchvision依赖FFmpeg，需要使用源码安装FFmpeg。或者使用pip安装 ``pip install torchvision==0.7.0`` 。

Q: 为什么无法运行Torchvision的自定义算子？

A: 如果希望使用Torchvision中的第三方算子，例如需要在MLU设备上运行NMS算子，此时需要安装与Cambricon PyTorch版本对应的Cambricon Torchvision。
若仅需要使用Torchvision的网络模型，可以使用 ``pip install torchvision==0.7.0`` 安装原生框架的Torchvision。

Q: Ubuntu16.04系统在AArch64平台交叉编译过程中出现报错：``fatal error: 'sys/cdefs.h' file not found``。

A: 执行命令 ``sudo apt-get install libc6-dev-i386`` 安装依赖项。

Q：使用MagicMind wheel包作为依赖进行编译和运行时出现失败，如何解决？

A：MagicMind wheel包目前只支持Python 3.7环境。在编译或运行时，可能会出现以下问题：

   - pandas安装问题：catch中的 ``requirements.txt`` 中指定pandas==0.22.0，但该版本在Python 3.7环境下可能无法兼容。
   
     要解决此问题，请安装高版本pandas，比如1.3.5。
   
   - MagicMind wheel包分为new-abi和old-abi。
   
     如果MagicMind wheel包是old-abi制作出来：
   
     运行时，需要额外把MagicMind wheel包安装路径和LLVM的依赖包路径加入到 ``LD_LIBRARY_PATH`` 中。

     .. code:: shell
     
        export LD_LIBRARY_PATH=YOUR_LOCAL_PATH/neuware_home/lib64:YOUR_LOCAL_PATH/neuware_home/lib/llvm-mm-cxx11-old-abi/lib/:/YOUR_PYTHON_PATH/site-packages/magicmind
   
     对于new-abi制作的MagicMind wheel包，运行时环境变量的设置方法同上。

Cambricon PyTorch融合推理
----------------------------

Q：原位 ``net = torch::jit::trace(net, example_forward_input)`` 后的net执行 ``net.to('mlu')`` 时，报如下错误：

   ::
   
     Traceback (most recent call last):
     File "test.py", line 20, in <module>
        net.to("mlu")
     File "venv/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 608, in to
        return self._apply(convert)
     File "venv/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 354, in _apply
        module._apply(fn)
     File "venv/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 382, in _apply
        assert isinstance(param, Parameter)
     AssertionError

A：由于trace后将net原位赋值，导致 ``net.parameters()`` 中param的类型由 ``torch.nn.parameter.Parameter`` 变为 ``torch.Tensor``，
进而导致 ``assert isinstance(param, Parameter)`` 失败。要解决该问题，使用非原位的方式执行trace，
即：``traced_net = torch::jit::trace(net, example_forward_input)``。

Q：为什么上述问题中使用原位方式trace时，执行 ``net.to('cpu')`` 未报错？

A：原生 ``to`` 使用以下接口控制是否覆盖Tensor：

   ::

     torch.__future__.set_overwrite_module_params_on_conversion(bool flag) 

由于原生CPU在执行 ``to`` 操作时，默认行为是直接替换Tensor中的TensorImpl结构，即不覆盖Tensor，因此未报错。
而MLU的MLUTensorImpl为TensorImpl子类，不能原位直接替换，而是使用覆盖Tensor的方式，因此会报错。

.. attention::

   | 目前Cambricon PyTorch仅支持 ``torch.__future__.set_overwrite_module_params_on_conversion(True)`` 条件下，CPU执行不报错的情况。

Q：使用 ``catch/script/release/independent_build.sh`` 脚本打patch出现告警：
   
   ::

      Warning: You have applied patches to Pytorch.

A：该告警说明用户已经打过patch到PyTorch，无需重复操作。
