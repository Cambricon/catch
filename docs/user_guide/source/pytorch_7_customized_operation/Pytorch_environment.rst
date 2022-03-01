配置环境变量
----------------------------
使用以下命令进行环境变量配置。

**DEBUG**

该环境变量用来设置debug模式。设置为1表示开启debug模式，设置为0表示禁止debug模式。

::

  export DEBUG=0

**USE_OPENCV**

该环境变量用来设置是否使用OpenCV。设置为1表示使用OpenCV，设置为0表示不使用。

**MAX_JOBS**

该环境变量用来设置编译最大Job数。

**USE_CUDA**

该环境变量用来设置是否使用CUDA。设置为0表示不使用CUDA，设置为1表示使用CUDA。

**ENABLE_CNNL_TRYCATCH**

该环境变量用来设置是否使能CNNL算子try catch功能。若使能，CNNL算子运行失败该算子会自动运行到CPU上。设置为0/off/OFF表示关闭，其他值表示使能。默认为使能。

**USE_MAGICMIND**

该环境变量用来设置CATCH编译时是否使能MagicMind算子后端。若使能，PyTorch JIT推理时可使用MagicMind后端进行图融合加速推理。设置为0/off/OFF表示关闭，其他值表示使能。默认为使能。

**ENABLE_MAGICMIND_FUSION_MODE**

该环境变量用来设置CATCH运行时是否使能MagicMind后端FUSION融合模式。若使能，PyTorch JIT推理时可使用MagicMind后端FUSION模式进行图融合加速推理。设置为0/off/OFF表示关闭，其他值表示使能。默认为使能。
