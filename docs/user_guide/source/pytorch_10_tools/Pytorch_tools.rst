.. _推理模型量化工具:

推理模型量化工具
----------------------

本节主要介绍如何使用模型量化工具生成量化的模型以及使用量化好的模型进行推理。

量化可以加快网络运行速度，降低占用的位宽。神经网络具有鲁棒性，如果训练的网络鲁棒，那么量化之后一般不会降低多少精度，有的甚至会提高精度。寒武纪软件栈针对卷积、全连接算子等必须要进行量化后才能运行；而其他如激活算子、BN算子等不需要量化，直接使用浮点型计算。

调用量化接口，会将需要量化的算子替换，Cambricon PyTorch会对以下列表中的算子进行替换。更多算子替换信息，参考 ``torch_mlu/core/quantized/default_mappings.py`` 。

+------------------------+----------------------------+
| 原生PyTorch算子        | 替换后的MLU算子            |
+========================+============================+
| nn.Linear              | cnq.MLULinear              |
+------------------------+----------------------------+
| nn.Conv2d              | cnq.MLUConv2d              |
+------------------------+----------------------------+

.. attention::

   -  ``nn``  是 ``torch.nn`` 的简写。
   -  ``cnq``  是 ``torch_mlu.core.quantized.modules`` 的简写。

使用量化接口
""""""""""""""""""""""""""

Catch集成的量化接口用于生成量化模型和运行量化模型。

量化接口原型如下：

::

  quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model, qconfig_spec=None, dtype=None, mapping=None, inplace=False, gen_quant=False)

**参数**

- model：待量化的模型。在生成量化模型时，model必须先加载原始权重，再调用该接口。在运行量化模型时，调用完该接口后，model再加载量化后的模型。

- qconfig_spec：配置量化的字典。

  默认为 ``{'use_avg': False, 'data_scale': 1.0, 'firstconv': False, 'mean': None, 'std': None, 'per_channel': False}`` 。

  - use_avg：设置是否使用最值的平均值用于量化。默认值为False，即不使用。该参数为超参数，调节该参数以获得一个比较好的输入张量的取值区间，进而得到合适的缩放尺度。

  - data_scale：设置是否对图片的最值进行缩放。默认值为1.0，即不进行缩放。该参数为超参数，调节该参数以获得一个比较好的输入张量的取值区间，进而得到合适的缩放尺度。

  - firstconv：设置是否使用firstconv。默认值为False，即不使用。当开启firstconv时，需要同时设置mean和std参数。 该功能将原始图片减均值除方差的预处理工作放在MLU设备上，从而尽可能的提升e2e性能。
  
  - mean：设置数据集的均值。默认值为None，取值范围为[0, 1)，实际计算时会乘以255。

  - std：设置数据集的方差。默认值为None，取值范围为(0, 1)，实际计算时会乘以255。

  - per_channel：设置是否使用分通道量化。默认值为False，即不使用分通道量化。

  - method：设置量化方法，当前只支持minmax量化方法。
  
- dtype：设置量化的模式。当前支持‘int8’和‘int16’模式，使用字符串类型传入。

- mapping：设置待量化算子。不指定该参数时，量化接口会按照以上表格进行算子替换；如果调用自定义算子，必须指定该参数。

- inplace：设置量化接口是否在原模型的基础上做原位操作。默认为False，表示先进行深拷贝再进行更改，否则就是在原模型的基础上做量化相关更改。

- gen_quant：设置是否生成量化模型。默认为False，表示不生成量化模型。在生成量化模型时，需设置为True；在运行量化模型时，需设置为False。

调用该量化接口后：

- 如果是生成量化模型，quantized_model中的可量化算子，比如Conv2d，会替换为MLUConv2d，且均加入了量化Observer和hook等信息，用于计算scale。
- 如果是运行量化模型，quantized_model中的可量化算子，比如Conv2d，会替换为MLUConv2d（量化信息在生成量化模型时已保存在权重中）。

生成量化模型
~~~~~~~~~~~~~~~~~~~~~~

本节以生成ResNet50量化模型为例介绍生成量化模型的步骤。

1. 获取ResNet50网络文件。

   ::

     net = resnet50()

2. 加载原始权重文件。

   ::

     state_dict = torch.load("path_origin_resnet50.pth")
     net.load_state_dict(state_dict)

3. 调用量化接口。

   ::

     quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(net, dtype='int8', gen_quant=True)

4. 在CPU上运行推理，生成量化值。

   ::

     quantized_model(input_tensor)

5. 保存量化模型。

   ::

     torch.save(quantized_model.state_dict(), "path_quantize_resnet50.pth")

运行量化模型
~~~~~~~~~~~~~~~~~

本节以运行ResNet50量化模型为例介绍运行量化模型的步骤。

1. 获取ResNet50网络文件。

   ::

     net = resnet50()

2. 调用量化接口以替换权重算子，例如：conv2d/linear。

   ::

     quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(net)

3. 加载量化权重文件。

   ::

     state_dict = torch.load("path_quantize_resnet50.pth")
     quantized_model.load_state_dict(state_dict)

4. 运行推理。

   - 在MLU上以CNNL后端进行推理

     ::
     
       quantized_model_mlu = quantized_model.to(ct.mlu_device())
       quantized_model_mlu(img.to(ct.mlu_device()))

   - 在MLU上以MagicMind后端进行推理

     ::
     
       quantized_model_mlu = quantized_model.to(ct.mlu_device())
       traced_model_mlu = torch.jit.trace(quantized_model_mlu, example_input.to(ct.mlu_device()), check_trace=False)
       traced_model_mlu(img.to(ct.mlu_device())

量化原理
""""""""""""""""""""""""""

本小节以int8量化为例，简要说明量化原理。

- **生成量化模型过程**

  量化模型是float数值向int数值映射的过程。
  
  以int8为例，要将浮点输入映射成定点，需要先统计输入的数值范围，得到其绝对值的最大值，记为absmax；然后将absmax映射到127（int8下的最大值），得到映射的缩放尺度scale为 ``scale = 127 / absmax``。该scale即浮点输入映射为定点的缩放尺度。同理，可以计算出权值的缩放尺度。将待量化算子的输入和权重的缩放尺度保存在模型的参数里，调用 ``torch.save`` 将其存储为 ``pth`` 文件，用于后续在MLU上运行定点计算。

- **运行量化模型过程**

  以卷积算子为例，首先对卷积算子的浮点输入进行量化，使用量化模型中的scale值，根据量化公式 ``qx = clamp(round(x * scale))`` 计算得到整型输入。同理，对该算子的权值进行量化，得到整型的权值；然后进行整型的卷积运算输出整型的卷积结果，根据反量化公式 ``y = qy / scale1 / scale2``，最终得到浮点的卷积输出。

  运行量化模型过程如下图所示：
  
  .. figure:: ../doc_image/quantization.png
  
     运行量化模型过程

量化不支持情况
""""""""""""""""""""""""""

目前不支持自定义算子量化。

例如：

::

  class IConv(torch.nn.Conv2d):
     def __init__(self):
       super(IConv, self).__init__()
     ...

这种属于自定义了IConv继承torch.nn.Conv2d，目前无法支持该种类型的量化，自定义的IConv会被识别为非量化函数。支持量化的class类必须是继承torch.nn.Module。

.. _调试工具:

调试工具
---------------------------------
为了便于用户调试和快速定位问题，Cambricon CATCH提供了日志打印工具供用户使用。

Cambricon CATCH通过设置以下环境变量来控制日志的打印级别，帮助用户定位和分析程序运行中的问题。

::

  export TORCH_MIN_CNLOG_LEVEL=LOG_LEVEL

Cambricon CATCH共有DEBUG(-1)、INFO(0)、WARNING(1)、ERROR(2)、FATAL(3)五个级别。设置好LOG_LEVEL后，日志级别大于等于该LOG_LEVEL的信息会打印到屏幕。默认打印WARNING及以上级别的信息。例如，若要打印CATCH中各算子的输入参数信息，可设置LOG_LEVEL 值为-1。

MagicMind后端同样提供环境变量 ``MM_CPP_MIN_LOG_LEVEL`` 来控制日志的打印级别，帮助用户定位和分析程序运行中的问题。

::

  export MM_CPP_MIN_LOG_LEVEL=MM_LOG_LEVEL
  # MM_CPP_MIN_LOG_LEVEL=0: enable INFO/WARNING/ERROR/FATAL
  # MM_CPP_MIN_LOG_LEVEL=1: enable WARNING/ERROR/FATAL
  # MM_CPP_MIN_LOG_LEVEL=2: enable ERROR/FATAL
  # MM_CPP_MIN_LOG_LEVEL=3: enable FATAL

.. _数据DUMP工具:

数据DUMP工具
---------------------------------

使用Cambricon CATCH的CNNL后端进行训练或推理时，可使用DUMP工具将计算过程的中间数据保存为文件，以便分析计算过程并定位问题。

DUMP工具的接口原型如下：

::

  torch_mlu.core.dumptool.Dumper(dump_dir="./dump", enable=True, use_cpu=False, level=0)

**参数说明**

- **dump_dir**

  数据的保存路径，默认为“./dump”。如果指定目录存在，DUMP工具将按序添加数字后缀，以避免覆盖现有数据。

- **enable**

  DUMP工具开关，默认为True，即开启状态。

- **use_cpu**
  启用CPU对比功能，默认为False，即关闭状态。启用CPU对比时，DUMP数据时将同时使用CPU对数据进行计算，并保存CPU结果。网络运行以MLU计算结果为准，CPU的比对结果仅保存到文件，并不参与后续的实际计算。该功能会忽略不支持的算子和在CPU上运行失败的算子。
  
  CPU对比功能不支持原位算子（后缀为下划线或为out的算子）。

- **level**

  DUMP数据中tensor的规模，默认为0。目前可选0、1、2三种级别。
  
  - 0：tensor只保存前10项。
  - 1：计算并保存tensor的绝对值之和。
  - 2：tensor数据完整保存。
  
.. attention::
   
   - 启用DUMP工具将消耗较长的时间用于IO拷贝，模型计算时间将大幅延长。如果启用了CPU对比功能，也会消耗额外的计算时间。
   - 如果使用level=2的完整保存模式，将会保存指定范围内的所有算子数据，这会占用大量磁盘空间，且花费较长时间写入数据。
   - DUMP工具进程独立，如果使用多进程（如DDP）进行计算，需要在子进程中分别调用Dumper。

DUMP工具使用
""""""""""""""""""""""""""
DUMP工具依据Python的上下文管理协议设计，推荐使用Python的上下文管理接口来调用该工具。

以下以pytorch_models中的分类网络为例。

.. code:: python

   # In function train:
   for i, (images, target) in enumerate(train_loader):
       from torch_mlu.core.dumptool import Dumper
       debug_iter=[0, 1, 2, 3]
       with Dumper(dump_dir=f"./dump_iter_{i}",
                   enable = (i in debug_iter),  # 配置enable参数，仅当i在0-3范围内时启用DUMP工具
                   use_cpu = True,
                   level = 2) :
           ... # input pre-processing
           output = model(images)
           loss = criterion(output, targe)
           loss.backward()
       # End of Dumper

上述代码将网络的前向、损失函数和反向的计算都放入DUMP工具的上下文管理范围内，并指定在前4个迭代进行DUMP，且运行CPU对比，并保存完整数据信息。

DUMP结果说明
""""""""""""""""""""""""""
运行代码时，会依次建立dump_iter_0到dump_iter_3的四个目录，并将对应迭代中所使用的Cambricon CATCH算子相关数据保存到相应目录下。

以ResNet 101为例，运行该网络保存的目录结构（部分）如下：

::

  dump_iter_0
  ├── 1_convolution_overrideable
  │   ├── cnnl_result
  │   ├── mlu_bias
  │   ├── mlu_input
  │   └── mlu_weight
  ├── 2_add
  │   ├── 1_fill_
  │   │   ├── cnnl_result
  │   │   ├── mlu_self
  │   │   └── mlu_value
  │   ├── 2_local_scalar_dense
  │   │   ├── cnnl_result
  │   │   └── mlu_self
  │   ├── mlu_alpha
  │   ├── mlu_other
  │   └── mlu_self
  ...


示例中，该网络首个调用的算子为convolution_overrideable，该算子使用mlu_bias、mlu_input、mlu_weight作为输入，计算结果为cnnl_result。第二个算子为add算子，该算子又分别调用了fill_算子和local_scalar_dense算子，每个算子的输入输出在各自目录保存。

保存的数据中包含类型、数据类型、形状等信息。以convolution_overrideable算子的cnnl_result（部分）为例：

::

  Tensor : Type = Float : Shape = [64, 64, 112, 112] : Dumped = 51380224
  -6.17937
  -7.91894
  -8.33218
  -8.35655
  -8.17265
  -7.86192
  -7.89231
  -8.27133
  ...

其中cnnl_result为tensor类型，数据类型为Float，形状为64×64×112×112。在本例中使用level=2的完整保存，因此将其全部数值共51380224个保存至文件。后续数值为tensor中各元素的值。

.. _MLU算子融合工具:

MLU算子融合工具
--------------------------------------------------

用来控制是否使用mlu的算子融合模式，以提高算子的效率。

.. attention::

  - 这里的融合模式是指使用一个大算子替换掉多个小算子拼接的过程。
  - 目前支持融合模式的算子有lstm。

.. code:: python

    import torch
    import torch_mlu
    #默认是True
    torch.backends.mlufusion.set_flags(False)
    ...
    #业务代码
    ...
    
    if torch.backends.mlufusion.enabled:
    　　...
        #业务代码
      　...
