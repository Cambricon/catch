PyTorch概述
===================

PyTorch是一款Facebook开源的深度学习编程框架，适用于Python、C++等编程语言，用以实现高效的GPU并行计算及深度学习网络搭建，具有轻松扩展、快速实现、生产部署稳定性强等优点。

PyTorch在Python中重新设计和实现Torch（一个用于机器学习和科学计算的模块化开源库。该库通过利用LuaJIT编译器提高性能），同时在后端代码中共享相同的核心C库。PyTorch开发人员优化了其后端代码来高效运行Python，同时还保留了其基于GPU的硬件加速和基于Lua的Torch的可扩展特性。PyTorch的后端不是单一的，根据不同的设备和功能可使用不同的后端，例如，处理CPU张量使用TH后端，处理GPU张量使用THC后端。同理，针对CPU和GPU的神经网络后端分别是THNN和THCUNN。

Python C扩展模块
::::::::::::::::::::

尽管PyTorch是Torch的衍生品，但它被特意设计为一个原生的Python包，其中的所有功能都构建成了Python类。因此，PyTorch代码能够与Python函数和其他Python包无缝集成。但是PyTorch的底层核心库是基于C/C++，因此需要依赖Python C扩展模型将Torch库中的Python接口与底层C/C++实现衔接起来。

ATen模块
::::::::::::::::::::

ATen模块（A Tensor Library）是PyTorch的tensor库，提供了很多张量操作，还实现了一些算子（例如卷积）在CPU端和GPU端的前向、后向计算。此外，上层算子也是在该模块下确定其应该调用哪个算子计算以及分发到哪个设备上运行。

代码自动生成模块
::::::::::::::::::::

PyTorch中有些C++代码是在编译PyTorch的过程中创建出来的，这个创建过程是由Python脚本完成。使用代码自动生成的原因有两个：一是可复用很多代码的逻辑；二是根据配置渲染模型。在扩展MLU后端时，新增接口、新增文件均需修改相应的模板配置，以便代码自动生成模块生成对应的C++代码。

关于原生PyTorch的更多内容，参见原生PyTorch官方文档。

Cambricon PyTorch概述
===============================

为支持寒武纪MLU（Machine Learning Unit，机器学习处理器），寒武纪定制了开源深度学习编程框架PyTorch（以下简称Cambricon PyTorch）。

Cambricon PyTorch借助PyTorch自身提供的设备扩展接口将MLU后端库中所包含的算子操作动态注册到PyTorch中，MLU后端库可处理MLU上的张量和神经网络算子的运算。Cambricon PyTorch会基于CNNL库和MagicMind库在MLU后端实现一些常用神经网络算子，并完成一些数据拷贝。

Cambricon PyTorch兼容原生PyTorch的Python编程接口和原生PyTorch网络模型，支持以在线逐层方式进行训练和以JIT融合方式进行推理。网络权重可以从pth格式文件读取，已支持的分类和检测网络结构由Torchvision管理，可以从Torchvision中读取。对于训练任务，支持float32及定点量化模型。对于推理任务，暂时只支持float32数据类型。

为了能在Torch模块方便使用MLU设备，Cambricon PyTorch在PyTorch后端进行了以下扩展：

- 通过Torch模块可调用MLU后端支持的神经网络运算。
- 对MLU暂不支持的算子，并且该算子在MLU后端库中已添加注册，支持该类算子自动切换到CPU上运行。
- Torch模块中与MLU相关的接口的语义与CPU和GPU的接口语义保持一致。
- 支持CPU和MLU之间的无缝切换。

从PyTorch 1.3.0开始，寒武纪采用Python扩展包的形式对原生PyTorch进行支持。寒武纪将所有关于MLU的操作都放在了一个单独的Python包中，然后将该包导入到原生PyTorch以支持在MLU上的运算。

本手册主要介绍了基于MLU的Cambricon PyTorch使用方法，这里的MLU适用于MLU370硬件版本。

