以基于ImageNet数据集的ResNet50模型训练迁移至MLU设备运行为例。

获取样例
---------------------

本样例基于PyTorch官网提供的 `ImageNet数据集模型训练脚本 <https://github.com/pytorch/examples/tree/master/imagenet>`_ 进行适配MLU 370处理器的迁移改造。

迁移训练脚本
---------------------

对照模型训练脚本，按照以下替换内容修改仓库中的 ``main.py`` 脚本。

.. list-table:: ResNet模型迁移接口替换-设备接口
    :widths: 65 65 60
    :header-rows: 1

    * - 原始程序
      - 修改后程序
      - 修改说明

    * - ``import torch``
      - ``import torch``

        ``import torch_mlu``
      - 增加torch_mlu依赖

    * - ``torch.cuda.device_count()``
      - ``torch.mlu.device_count()``
      - 获取MLU设备数量
    
    * - ``torch.cuda.is_available():``
      - ``torch.mlu.is_available():``
      - 判断MLU设备是否可用
   
    * - ``torch.cuda.set_device``

        ``model.cuda(args.gpu)``
      - ``torch.mlu.set_device``

        ``model.mlu(args.gpu)``
      - 模型切换至指定MLU设备


.. list-table:: ResNet模型迁移接口替换- 模块调用
    :widths: 70 70 50 
    :header-rows: 1

    * - 原始程序
      - 修改程序
      - 修改说明
  
    * - ``model.cuda()``
      - ``model.mlu()``
      - 模型切换至MLU设备

    * - ``nn.CrossEntropyLoss()``
            
        ``.mlu(args.gpu)``
      - ``nn.CrossEntropyLoss()``

        ``.mlu(args.gpu)``
      - 修改Loss层至MLU设备

    * - ``'cuda:{}'.format(args.gpu)``
      - ``'mlu:{}'.format(args.gpu)``
      - 设备字符串修改为MLU设备

    * - ``images.cuda(args.gpu, non_blocking=True)``
          
        ``torch.cuda.is_available()``
        
        ``target.cuda(args.gpu, non_blocking=True)``
        
      - ``images.mlu(args.gpu, non_blocking=True)``
        
        ``torch.mlu.is_available()``
        
        ``target.mlu(args.gpu, non_blocking=True)``
      - 将输入数据切换至MLU设备

脚本执行
---------------------

准备数据集
'''''''''''''''

下载 `ImageNet数据集 <http://www.image-net.org/>`_ 保存至运行环境目录下，例如: ``/data/imagenet``。

模型单卡训练
'''''''''''''''

.. code:: shell

   python main.py /data/imagenet --gpu 0 --batch-size 128 --lr 0.1 --epochs 90 --arch resnet50 --workers 40 --momentum 0.9 --weight-decay 1e-4


模型多卡训练
'''''''''''''''

.. code:: shell

   python main.py /data/imagenet --gpu 0 --batch-size 128 --lr 0.1 --epochs 90 --arch resnet50 --workers 40 --momentum 0.9 --weight-decay 1e-4 --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:65501' --world-size 1 --rank 0 --dist-backend 'cncl'

.. attention::
   | 此处，需将参数 ``dist-backend`` 值修改为 ``'cncl'``。
  
