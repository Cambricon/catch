本章介绍Cambricon PyTorch融合推理的API运行模式以及使用方法。

支持的典型网络
----------------

.. tabularcolumns:: |m{0.2\textwidth}|m{0.7\textwidth}|
.. table:: 支持的典型网络

   ========= ========================================================================
   类别      名称
   ========= ========================================================================
   分类网络  VGG19、ResNet50
   ========= ========================================================================

.. attention::


   - 运行推理要求GCC 7，且编译时设置环境变量为 ``USE_MAGICMIND=ON``。
   - 目前只支持torch.jit.trace得到的融合模型推理，支持float32和float16。
   - 目前MagicMind推理后端仅支持Conv2d/Linear算子的int8/int16量化。
   - 目前MagmicMind量化方式仅支持全tensor，对称量化。后续会扩展更多量化方式。

.. _Python API使用:

Python API使用
------------------
Cambricon PyTorch不改变原生PyTorch的接口行为，在CATCH扩展包中添加MLU设备以及MLU算子。通过在原生PyTorch上打补丁实现了PyTorch的大部分特性。

Cambricon PyTorch支持CPU和MLU设备类型。可以使用 ``to`` 方法将CPU设备上的tensor以及script module转为MLU对象。

Cambricon PyTorch使用MagicMind推理后端支持PyTorch script module的融合模式推理。提供接口 ``_jit_override_can_fuse_on_mlu(bool flag)`` 使能与关闭融合推理
，默认为使能。

使用方式如下：

.. code:: python

   import torch_mlu.core.mlu_model as ct
   # 关闭MagicMind融合推理模式。
   ct._jit_override_can_fuse_on_mlu(False)

Cambricon PyTorch融合推理使用原生PyTorch JIT 推理API。以下分别从单算子和网络角度介绍如何使用PyTorch JIT API进行前向推理。

使用Python API执行单算子推理
""""""""""""""""""""""""""""""

以下介绍Cambricon PyTorch在脚本上如何部署单算子推理。

权重算子
~~~~~~~~~~~~~~~~~~~~~~

以卷积算子为例，使用PyTorch JIT API运行推理如下：

.. code:: python

   import torch
   import torch_mlu
   import torch.nn as nn
   import torch_mlu.core.mlu_model as ct

   # 默认为使能。如果没有更改过，略去以下语句。
   ct._jit_override_can_fuse_on_mlu(True)
   # 执行推理必须设置以下条件：
   torch.set_grad_enabled(False)

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv = nn.Conv2d(1, 1, 3)

       def forward(self, x):
           return self.conv(x)

   n = Net().eval().float()
   example_forward_input = torch.rand(1, 1, 3, 3)

   module = torch.jit.trace(n, example_forward_input)
   input = torch.randn((1,1,28,28), dtype=torch.float)
   input_mlu = input.to("mlu")
   module.to("mlu")
   out_mlu = module(input_mlu)
   out_cpu = out_mlu.cpu()

以上代码展示了如何使用MLU设备完成卷积运算。其中，``input.to("mlu")`` 操作实际上是将input转移至MLU。

非权重算子
~~~~~~~~~~~~~~~~~~~~~~
以激活算子 ``ReLU`` 为例，使用PyTorch JIT API运行推理如下：

.. code:: python

   import torch
   import torch_mlu
   torch.set_grad_enabled(False)
   input = torch.randn(4,10,dtype=torch.float)
   relu = torch.nn.ReLU().float().eval()
   module = torch.jit.trace(relu, input)
   relu_mlu = module.to("mlu")
   out_mlu = relu_mlu(input.to("mlu"))
   out_cpu = out_mlu.cpu()

以上代码展示了如何使用MLU设备完成ReLU激活操作。

使用Python API执行网络推理
""""""""""""""""""""""""""""""
目前MigicMind推理后端支持以下数据类型：

- 非权重算子：float16、float32。

- 权重算子（Conv2d/Linear）：int8、int16、float16、float32。

以下介绍如何使用Python API在MagicMind后端对网络执行不同数据类型的推理。

首先，自定义网络模型 ``TestModel``。之后的推理脚本以该模型为例。

.. code:: python

   import torch
   import torch.nn as nn
   import torch_mlu
   import torch_mlu.core.mlu_model as ct
   from torch.nn import Parameter
   import torch.nn.functional as F
   import random
   # 执行推理必须设置以下条件：
   torch.set_grad_enabled(False)

   class TestModel(nn.Module):
       def __init__(self, in_channels):
           super(TestModel, self).__init__()
           out_channels = 16
           conv1 = torch.nn.Conv2d(in_channels, out_channels, \
                                   3, 1, 0, 1, 1, bias=False)
           bn1 = torch.nn.BatchNorm2d(out_channels, affine=False)
           relu1 = torch.nn.ReLU(inplace=False)
           self.block1 = torch.nn.Sequential(conv1,
                                             bn1,
                                             relu1)

       def forward(self, x):
           y = self.block1(x)
           z = y + y
           z1 = F.max_pool2d(z, 2, stride=None)
           z2 = torch.transpose(z1, 0, 3)
           return z2

   # 初始化模型，并准备网络的输入数据。
   in_channels = random.randint(1,10)
   input1 = torch.rand(1, in_channels, 224, 224)
   model = TestModel(in_channels)
   model.eval().float()

然后，根据不同数据类型执行网络推理。

- 使用float32数据类型执行网络 ``TestModel`` 的推理

  .. code:: python
  
     traced_model = torch.jit.trace(model.to(torch.float32), input1.to(torch.float32), check_trace=False)
     traced_model.to(ct.mlu_device())
     output = traced_model(input1.to('mlu'))

- 使用float16数据类型执行网络 ``TestModel`` 的推理

  .. code:: python
  
     traced_model = torch.jit.trace(model, input1, check_trace=False)
     # 将权重数据转为float16类型。
     traced_model.half().to('mlu')
     # 网络的输入数据也需要转为float16类型，并送入模型进行前向计算。
     output = traced_model(input1.half().to('mlu'))

  其中，``traced_model.half()`` 将模型中的权重数据类型转换为float16， ``input1.half()`` 将输入数据类型转换为float16。

- 使用int8（权重算子）_float32（非权重算子）数据类型执行网络 ``TestModel`` 的推理

  .. code:: python
  
     # 首先，导入量化工具包。
     import torch_mlu.core.mlu_quantize as mlu_quantize
     
     # 初始化量化模型，以便在前向时计算输入和权重的量化参数，dtype设置为int8。
     qconfig = {'use_ave': False, 'data_scale': 1.0, 'mean': None, 'std': None, 'firstconv': False}
     quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8', gen_quant=True)
     
     # 在cpu上执行前向函数，输入和权重的量化参数会被保存到模型的权重字典：state_dict()中。
     example_input = torch.rand(1, in_channels, 224, 224)
     _  = quantized_model(example_input)
     checkpoint = quantized_model.state_dict()
  
     # 部署并运行量化后的模型。
     model_mlu =  mlu_quantize.quantize_dynamic_mlu(model)
     # 加载包含量化参数的权重
     model_mlu.load_state_dict(checkpoint)
     # 注意，这里的model_mlu中的权重算子已被替换为MLU量化权重算子。
     # 所以使用jit.trace时需要将输入和权重的设备类型设置为‘mlu’。
     traced_model = torch.jit.trace(model_mlu.to('mlu'), input1.to('mlu'), check_trace=False)
     output = traced_model(input1.to('mlu'))

- 使用int16（权重算子）_float16（非权重算子）数据类型执行网络 ``TestModel`` 的推理

  .. code:: python
  
     import torch_mlu.core.mlu_quantize as mlu_quantize
     
     # 初始化量化模型，以便在前向时计算输入和权重的量化参数，dtype需设置为int16。
     qconfig = {'use_ave': False, 'data_scale': 1.0, 'mean': None, 'std': None, 'firstconv': False}
     quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int16', gen_quant=True)
     
     # 在cpu上执行前向函数，输入和权重的量化参数会被保存到模型的权重字典：state_dict()中。
     example_input = torch.rand(1, in_channels, 224, 224)
     _  = quantized_model(example_input)
     checkpoint = quantized_model.state_dict()
  
     # 部署并运行量化后的模型。
     model_mlu =  mlu_quantize.quantize_dynamic_mlu(model)
     # 加载包含量化参数的权重
     model_mlu.load_state_dict(checkpoint)
  
     # 注意，这里的model_mlu中的权重算子已被替换为MLU量化权重算子。
     # 所以使用jit.trace时需要将输入和权重的设备类型设置为'mlu'。
     traced_model = torch.jit.trace(model_mlu.to('mlu'), input1.to('mlu'), check_trace=False)
     traced_model.half()
     output = traced_model(input1.half().to('mlu'))

以上代码展示了如何使用Python API在MLU设备上完成float32、float16、float32_int8、float16_int16的推理部署。float16_int8和float32_int16的推理部署也可参考上述代码来实现。

关于量化接口 ``mlu_quantize.quantize_dynamic_mlu`` 的详细说明，请参考 :ref:`推理模型量化工具` 小节。

融合模式推理
----------------

融合模式推理
"""""""""""""""""""

融合模式推理指使用原生PyTorch提供的JIT API直接运行网络。

融合模式将MagicMind支持的算子融合为一个或多个fusion算子，只对fusion算子执行编译指令过程，减少了小算子之间的
数据拷贝（不仅是主从设备间，还包括RAM和DDR之间的拷贝），极大地提高了效率。使用JIT模式只需对整个网络进行
一次编译，避免了多次编译产生的开销。

算子运行Fallback功能
""""""""""""""""""""""""

融合模式推理使用MagicMind后端，根据MagicMind支持的算子情况使用JIT pass融合为一个或多个fusion算子（一个fusion算子对应一个由MagicMind算子组成的融合子图）交由MagicMind后端执行。
MagicMind不支持的算子优先fallback到CNNL后端执行，
对于CNNL目前不支持的算子会fallback到CPU执行。（fallback到CPU功能，参见 :ref:`MLU未实现算子自动运行到CPU` 章节）。
对于MagicMind支持的算子，可以通过设置环境变量 ``DEBUG_FORCED_FALLBACK_OPS`` 将算子设置为黑名单，强制将这些算子fallback到CNNL后端执行。

算子调试与测试
"""""""""""""""""""

Cambricon Catch提供了对fusion算子结果的调试功能，使用环境变量 ``FUSED_KERNEL_DEBUG=cnnl`` 或 ``cpu``，
保存fusion融合算子在MagicMind后端的运行结果到 ``mmouttensor_MLUFusionGroupi_j`` 文件，
同时根据环境变量设置，将fusion融合算子在CNNL后端或CPU的运行结果保存到 ``catch_cnnl_jitouttensor_MLUFusionGroupi_j`` 或 
``cpu_jitouttensor_MLUFusionGroupi_j`` 文件（其中 ``i`` 表示fusion算子的计数， ``j`` 表示当前fusion算子的输出计数）。

设置 ``TORCH_MIN_CNLOG_LEVEL=-1`` 调试模式，会将图分割后的带有融合算子的jit图打印出供调试使用。更多内容，参见 :ref:`调试工具` 章节。

在Cambricon CATCH中已经添加了常用网络需要的MLU算子。每个MLU算子都分别用单个文件添加了对该算子的测试。
每个文件均可使用Python直接运行并测试。

如果要对所有MLU算子进行测试，运行 ``<catch>/test/magicmind/op_test/test_all_operators.py`` 脚本。

.. code:: shell

   python test_all_operators.py

如果要对单个算子进行测试，执行该算子的对应脚本。例如对add算子进行测试，运行：

.. code:: shell

   python test_add.py

如果不需要测试某个函数，在该函数顶部加上 ``@unittest.skip("not test")`` 即可。
