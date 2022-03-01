开源PyTorch支持CPU和CUDA设备计算。如果使用Cambricon PyTorch在MLU设备上运行，需要将原始PyTorch模型脚本修改为MLU支持脚本。

Cambricon PyTorch支持MLU设备接口与CUDA接口相对应，将PyTorch CUDA模型脚本中与设备相关的接口修改为MLU设备接口支持，可完成PyTorch模型在MLU设备上开发、训练以及调试。

针对模型训练，主要迁移改动介绍如下：
 
单卡训练模型
---------------------

PyTorch模型单卡训练的主要改动包括：

导入torch_mlu模块
'''''''''''''''''''

.. code:: python

    import torch
    import torch_mlu

切换模型设备
''''''''''''

以ResNet50为例，模型处理包括以下步骤：

1. 定义模型。

   .. code:: python

      model = models.__dict__["resnet50"]()

2. 将模型加载到MLU上。

   .. code:: python

      mlu_model = model.to('mlu')


定义损失函数
''''''''''''''''

定义损失函数，然后将其拷贝至MLU。

.. code:: python

   # 构造损失函数
   criterion = nn.CrossEntropyLoss()
   # 将损失函数拷贝到MLU上
   criterion.to('mlu')

切换输入数据设备
'''''''''''''''''''

将数据从CPU拷贝到MLU设备。

.. code:: python

   x = torch.randn(1000000, dtype=torch.float)
   x_mlu = x.to(torch.device('mlu'), non_blocking=True)


有关模型在MLU设备运行单卡训练脚本的更多详细内容，参见样例 ``catch/examples/training/single_card_demo.py``。


多卡训练模型
---------------------

Cambricon PyTorch支持原生分布式相关功能。更多内容，参见《寒武纪PyTorch用户手册》的“模型训练”章节的分布式训练内容。

将原生PyTorch分布式相关脚本移植到MLU设备上，仅需修改以下内容：

替换通信后端
'''''''''''''''

MLU设备间通信依赖MLU设备通信后端CNCL（Cambricon NeuWare Communications Library，寒武纪通信库）。

初始化进程组实例时，``backend`` 参数只传入 ``cncl`` 。

.. code:: python

   # 创建进程组
   dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                           rank=rank, world_size=world_size)


有关模型在MLU设备运行分布式训练相关脚本的更多详细内容，参见样例 ``catch/examples/training/multi_card_demo.py``。
