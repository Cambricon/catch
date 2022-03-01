.. _模型迁移工具:

模型脚本转换工具
---------------------

从GPU模型脚本迁移至MLU设备运行，模型脚本修改位置较多。
使用模型脚本转换工具 ``torch_gpu2mlu.py`` 可对模型脚本进行转换，并对修改位置进行统计，实现开发者快速迁移。

.. attention::

   脚本转换工具会根据 :ref:`API支持列表` 进行修改转换。部分模型仍需要用户按照脚本实际情况进行少量适配。

本工具已针对以下开源模型完成测试。

**模型测试列表**

.. list-table:: 脚本转换工具模型测试列表
    :widths: 40 40 40
    :header-rows: 1

    * - 类别
      - 模型名称
      - 数据集

    * - 图像分类
      - `ResNet50 <https://github.com/pytorch/examples/tree/master/imagenet>`_
      - ImageNet2012


执行脚本转换工具
*********************

模型转换工具 ``torch_gpu2mlu.py`` 位于 ``catch/tools`` 目录下。

**参数介绍**

-i：指定模型脚本路径。

**使用示例**

.. code:: shell

   python <catch>/tools/torch_gpu2mlu.py -i <模型脚本路径>

输出转换结果
*********************

脚本执行后，终端会显示原始脚本路径、转换后脚本路径、以及模型脚本修改日志。

其中，转换后脚本和模型脚本修改日志均位于原始脚本相同路径的 ``_mlu`` 结尾的文件夹中。

以PyTorch ImageNet官方示例程序为例，使用该工具进行脚本转换，显示结果如下：

.. code:: shell

    $ python <catch>/tools/torch_gpu2mlu.py -i /tmp/imagenet
    $ Official PyTorch model scripts: /tmp/imagenet
    $ Cambricon PyTorch model scripts: /tmp/imagenet_mlu
    $ Migration Report: /tmp/imagenet_mlu/report.md

修改和运行脚本
*********************

以模型脚本修改日志为参照，对脚本进行修改、模型测试和验证。

脚本修改日志包含修改文件、修改位置对应的行号和修改内容。迁移过程中可以参考该日志进行模型脚本修改。

脚本修改日志示例如下：

.. code:: shell

    # Cambricon PyTorch Model Migration Report
    ## Cambricon PyTorch Changes
    | No. |  File  |  Description  |
    | 1 | main.py:8 | add "import torch_mlu" |
    | 3 | main.py:139 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
    | 4 | main.py:146 | change "torch.cuda.set_device(args.gpu)" to "torch.mlu.set_device(args.gpu) " |
    | 5 | main.py:147 | change "model.cuda(args.gpu)" to "model.mlu(args.gpu) " |
    | 6 | main.py:155 | change "model.cuda()" to "model.mlu() " |
    | 7 | main.py:160 | change "torch.cuda.set_device(args.gpu)" to "torch.mlu.set_device(args.gpu) " |
    | 8 | main.py:161 | change "model = model.cuda(args.gpu)" to "model = model.mlu(args.gpu) " |
    | 9 | main.py:166 | change "model.cuda()" to "model.mlu() " |
    
    ... ...
