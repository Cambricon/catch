模型执行
------------------

Q：多卡模型计算时，报错 ``"RuntimeError: Distributed package doesn't have NCCL built in"``，如何处理？

A：MLU设备只支持CNCL通信后端，因此需要修改脚本，将 ``init_process_group`` 接口中的 ``backend`` 参数为 ``'cncl'``。

