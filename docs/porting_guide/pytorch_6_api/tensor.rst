Tensor 数据类型
===================
.. list-table:: Tensor数据类型支持
    :widths: 30 45 45 45
    :header-rows: 1

    * - dtype
      - CPU Tensor
      - GPU Tensor
      - MLU Tensor 

    * - ``torch.float32``
      - ``torch.FloatTensor``
      - ``torch.cuda.FloatTensor``
      - ``torch.mlu.FloatTensor``

    * - ``torch.float16``
      - ``torch.HalfTensor``
      - ``torch.cuda.HalfTensor``
      - ``torch.mlu.HalfTensor``

    * - ``torch.uint8``
      - ``torch.ByteTensor``
      - ``torch.cuda.ByteTensor``
      - ``torch.mlu.ByteTensor``
    
    * - ``torch.int8``
      - ``torch.CharTensor``
      - ``torch.cuda.CharTensor``
      - ``torch.mlu.CharTensor``
    
    * - ``torch.int16``
      - ``torch.ShortTensor``
      - ``torch.cuda.ShortTensor``
      - ``torch.mlu.ShortTensor``
    
    * - ``torch.int32``
      - ``torch.IntTensor``
      - ``torch.cuda.IntTensor``
      - ``torch.mlu.IntTensor``
    
    * - ``torch.int64``
      - ``torch.LongTensor``
      - ``torch.cuda.LongTensor``
      - ``torch.mlu.LongTensor``
    
    * - ``torch.bool``
      - ``torch.BoolTensor``
      - ``torch.cuda.BoolTensor``
      - ``torch.mlu.BoolTensor``

    
Tensor相关接口替换
===================

.. list-table:: Tensor创建接口支持列表
    :widths: 20 70 70 
    :header-rows: 1

    * - 序号
      - CUDA设备接口
      - MLU设备接口
    
    * - 1
      - ``torch.tensor([0,1]).cuda()``
      - ``torch.tensor([0,1]).mlu()``

    * - 2
      - ``torch.tensor([0,1]).to('cuda')``
      - ``torch.tensor([0,1]).to('mlu')``

    * - 3
      - ``torch.tensor([0,1]).to('cuda:1')``
      - ``torch.tensor([0,1]).to('mlu:1')``

    * - 4
      - ``torch.tensor([0,1]).cuda().is_cuda``
      - ``torch.tensor([0,1]).mlu().is_mlu``

    * - 5 
      - ``torch.tensor([0,1], device='cuda')``
      - ``torch.tensor([0,1], device='mlu')``
