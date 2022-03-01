框架移植概述
------------
为了使Cambricon PyTorch更好地在寒武纪硬件平台上运行，寒武纪对原生PyTorch做了进一步拓展，如下图所示。Cambricon PyTorch对原生PyTorch的主要修改有：添加MLU设备、部分算子的分发方式、Torchvision访问权限和CATCH拓展包。有关Cambricon PyTorch的CATCH模块对PyTorch的修改信息和功能，见下表。

.. figure:: ../doc_image/catch_sf_framework.*

   Cambricon PyTorch 框架结构图

有关Cambricon PyTorch的CATCH模块对PyTorch的修改信息和功能，见以下表格。

.. tabularcolumns:: |m{0.1\textwidth}|m{0.55\textwidth}|m{0.25\textwidth}|
.. table:: CATCH对原生PyTorch的修改

   +------------+--------------------------------------------------+---------------------------------------------------------------+
   | 功能单元   | 修改文件                                         | 修改说明                                                      |
   +============+==================================================+===============================================================+
   | 设备支持   | c10/core/Backend.h                               | 添加对MLU设备的支持。                                         |
   |            |                                                  |                                                               |
   |            | c10/core/Device.cpp                              |                                                               |
   |            |                                                  |                                                               |
   |            | c10/core/DeviceType.cpp                          |                                                               |
   |            |                                                  |                                                               |
   |            | c10/core/DeviceType.h                            |                                                               |
   |            |                                                  |                                                               |
   |            | c10/core/TensorTypeId.cpp                        |                                                               |
   |            |                                                  |                                                               |
   |            | c10/core/TensorTypeId.h                          |                                                               |
   |            |                                                  |                                                               |
   |            | torch/csrc/utils/tensor_layouts.cpp              |                                                               |
   |            |                                                  |                                                               |
   |            | core/TensorOptions.h                             |                                                               |
   |            |                                                  |                                                               |
   |            | torch/tensor.py                                  |                                                               |
   +------------+--------------------------------------------------+---------------------------------------------------------------+
   | 兼容支持   | torch/hub.py                                     | 修改访问Torchvision权限。                                     |
   |            |                                                  |                                                               |
   +------------+--------------------------------------------------+---------------------------------------------------------------+
   | 算子分发   | aten/src/ATen/native/AdaptiveAveragePooling.cpp  | 修改部分算子分发方式。                                        |
   |            |                                                  |                                                               |
   |            | aten/src/ATen/native/Pooling.cpp                 |                                                               |
   +------------+--------------------------------------------------+---------------------------------------------------------------+

.. tabularcolumns:: |m{0.2\textwidth}|m{0.7\textwidth}|
.. table:: CATCH功能

   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | 模块                | 功能说明                                                                                                          |
   +=====================+===================================================================================================================+
   | Init模块            | 初始化CATCH库并将MLU算子、图分段算法以及图优化算法注册到PyTorch中。                                               |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | Python/C++接口      | 封装了一些Python转C++调用的接口来提供一些拓展功能。                                                               |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | ATen MLU后端        | 包括MLU Operator Register和MLU Operator Wrapper组件。                                                             |
   |                     | MLU Operator Register模块会调用PyTorch中的相关函数将MLU算子注册到PyTorch中。                                      |
   |                     | MLU Operator Wrapper用来封装MLU算子。                                                                             |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | JIT MLU后端         | MLU Fuser Operator、图分段算法以及图优化算法。                                                                    |
   |                     | MLU Fuser Operator用于完成MLU融合算子的创建、编译和运行。                                                         |
   |                     | 图分段算法会根据MagicMind算子支持情况将图中的节点划分成若干个子网络段。                                           |
   |                     | 图优化算法会对PyTorch IR图中的节点进行优化。                                                                      |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | distributed MLU后端 | 包括分布式训练的C++后端代码。                                                                                     |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | MLU Operators       | 调用CNNL算子或MagicMind算子来实现MLU算子的计算功能。                                                              |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | Patches for Pytorch | 对PyTorch的修改。                                                                                                 |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+
   | Demos               | 网络demo程序。                                                                                                    |
   +---------------------+-------------------------------------------------------------------------------------------------------------------+

.. _自定义在线逐层算子:

自定义在线逐层算子
--------------------

.. _添加逐层算子:

添加逐层算子
~~~~~~~~~~~~~~~~~~~~

PyTorch逐层模式中算子间数据传递和存储的基本单元是tensor。PyTorch根据tensor中的device属性值将算子分发到不同设备。本文以 ``abs()`` 算子为例，在dispatch阶段会根据input_tensor的设备属性值将算子调用分发到具体设备，如下图所示。

.. figure:: ../doc_image/mlu_operators.*

   逐层算子分发图

CATCH通过注册添加MLU算子方式与PyTorch源码解耦。

执行以下步骤在CATCH中添加MLU算子：

1. 声明算子。

   在 ``catch/torch_mlu/tools/mlu_functions.yaml`` 中声明算子。

   ::  

     - name: add # 算子名称
       use_mlu_dispatcher: unboxed_only # 分发类型，包括标准算子（unboxed_only）和自定义算子（custom）
       derived_type: cnnl # 派生类型（cnnl/bang）
       schema_string: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor # 用于算子注册的签名
       arguments: # 参数
       - name: self # 参数名称
         type: const at::Tensor & # 参数类型
       - name: other
         type: const at::Tensor &
       - name: alpha
         type: at::Scalar
       return_type: at::Tensor # 函数返回类型

   其中 ``derived_type`` 表示可以分发到寒武纪计算库CNNL或者分发到CATCH使用BANG C编写的自定义算子；
   ``schema_string`` 表示算子注册的签名，标准算子的签名必须与原生PyTorch中的签名一致，更多信息，参见 ``pytorch/torch/csrc/autograd/generated/RegistrationDeclarations.h`` 或者
   ``pytorch/aten/src/ATen/native/native_functions.yaml`` ；自定义算子的签名只需写明命名空间和函数名，如 ``torch_mlu::crop_resize`` 。

2. 添加OpMethods基类中的CPU实现。
   CATCH模块中包含AtenMluType标准算子类型和AtenMluCustomType自定义算子类型。AtenMluType和AtenMluCustomType通过OpMethods基类下发到不同计算库中的对应算子。
   OpMethods基类的作用是当MLU算子运行发生异常时，将MLU Tensor拷贝回CPU计算，再将计算结果重新拷贝回MLU进行后续计算。
   
   根据模板生成的op_methods.h头文件，在 ``catch/torch_mlu/csrc/aten/operators/op_methods.cpp`` 中添加算子的CPU实现。

   ::  
   
     op_methods.h
     virtual at::Tensor add(const at::Tensor& self, const at::Tensor& other, at::Scalar alpha);

   ::
   
     op_methods.cpp
     at::Tensor OpMethods::add(const at::Tensor& self,
                               const at::Tensor& other,
                               at::Scalar alpha) {
       auto input_cpu = self.cpu();
       auto other_cpu = other.cpu();
       auto output = at::add(input_cpu, other_cpu, alpha);
       return output.to(at::Device(at::Device::Type::MLU));
     }

   .. attention::
   
      | 对于新增的算子在执行过程中抛出异常时，如果OpMethods中未实现该算子的CPU操作，那么该算子无法切换到CPU上运行。

3. 添加wrapper。

   wrapper是对算子kernel的封装。每个算子对应一个wrapper。以下以add、div和dump算子为例，介绍不同计算库添加wrapper的方式：

   - CNNL算子

     ::

       cnnl_kernel.h
       at::Tensor cnnl_div(const at::Tensor& input, const at::Tensor& other);

     ::

       div.cpp
       at::Tensor cnnl_div(const at::Tensor& self, const at::Tensor& other) {
         at::Tensor input_new, other_new;
         bool input_is_scalar = false, other_is_scalar = false;
         if (self.dim() == 0 && other.dim() == 0) {
           auto self_t = self.cpu();
           auto other_t = other.cpu();
           auto output = at::div(self_t, other_t);
           return output.to(at::Device(at::Device::Type::MLU));
         } else if (other.dim() == 0) {
           // self is Tensor, other is Scalar
           auto other_tensor = at::native::full(
               self.sizes(), other.item(), self.options().device(at::kCPU));
           other_new = other_tensor.to(at::Device(at::Device::Type::MLU));
           other_is_scalar = true;
         }
         input_new = input_is_scalar ? input_new : self;
         other_new = other_is_scalar ? other_new : other;
         return cnnl_div_internal(input_new, other_new); //调用kernel
       }

   - BANG C自定义算子

     ::

       bang_kernel.h
       bool bang_dump(const at::Tensor & input);

     ::

       dump.cpp
       bool bang_dump(const at::Tensor& input) {
           auto input_impl = getMluTensorImpl(input);
           auto input_ptr = input_impl->cnnlMalloc();
           int32_t size = input.numel();
           cnrtDataType_t cnrt_type = fromCnnlType2CnrtType(input_impl->getCnnlType());
           cnrtDim3_t dim = {1, 1, 1};
           cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;
           auto queue = getCurQueue();
           dump(input_ptr, size, dim, ktype, queue, cnrt_type); // 调用BANG C kernel
           cnrtSyncQueue(queue);
           return true;
       }

   .. attention::
   
      | Wrapper一般以“cnnl_算子名”命名。

4. 添加kernel。

   Wrapper中通过调用kernel实现算子功能。示例中分别为CNNL库div算子以及CATCH中自定义BANG C算子dump算子。

   算子的具体实现主要通过调用CNNL或者BANG C接口来完成。不同库的逻辑如下：

   - CNNL库kernel

     CNNL库的kernel无需经过创建、编译、执行等步骤，使用相对简单，但不支持融合操作。
     
     在 ``catch/torch_mlu/csrc/aten/operators/cnnl/internal/cnnl_internal.h`` 和 ``catch/torch_mlu/csrc/aten/operators/cnnl/internal/div_internal.cpp`` 
     中分别添加kernel函数的声明和实现。

     ::

       cnnl_internal.h
       at::Tensor cnnl_div_internal(const at::Tensor& self, const at::Tensor& other);

     ::

       div_internal.cpp
       at::Tensor cnnl_div_internal(const at::Tensor& self, const at::Tensor& other) {
         at::Tensor input_new = self;
         at::Tensor other_new = other;
         auto output = at::empty_like(self);

         auto input_impl = getMluTensorImpl(input_new);
         auto other_impl = getMluTensorImpl(other_new);
         auto output_impl = getMluTensorImpl(output);

         // 获取当前句柄
         auto handle = getCurrentHandle();
         auto queue = getCurQueue();
         CnnlTensorDescriptor desc_input;
         CnnlTensorDescriptor desc_other;
         CnnlTensorDescriptor desc_output;

         // 设置参数描述符
         desc_input.set(input_new);
         desc_other.set(other_new);
         desc_output.set(output);

         // 分配MLU内存
         auto input_ptr = input_impl->cnnlMalloc();
         auto other_ptr = other_impl->cnnlMalloc();
         auto output_ptr = output_impl->cnnlMalloc();

         // 获取需要额外分配的workspace大小
         size_t workspace_size = 0;
         TORCH_CNNL_CHECK(cnnlGetDivWorkspaceSize(handle,
                               desc_input.desc(),
                               desc_other.desc(),
                               desc_output.desc(),
                               &workspace_size));
         std::vector<int64_t> space_shape;
         workspace_size /= input_impl->itemsize();
         at::Tensor temp =
             at::empty({static_cast<long int>(workspace_size)}, self.options());
         auto* temp_impl = getMluTensorImpl(temp);
         auto temp_ptr = temp_impl->cnnlMalloc();

         // 调用CNNL kernel
         TORCH_CNNL_CHECK(cnnlDiv(handle, desc_input.desc(), input_ptr,
                                  desc_other.desc(), other_ptr,
                                  temp_ptr, desc_output.desc(), output_ptr));
         TORCH_CNRT_CHECK(cnrtSyncQueue(queue));
         return output;
       }

   - BANG C自定义算子
   
     跟CNNL库相同的是，CATCH中的BANG C自定义算子也无需经过创建、编译、执行等步骤，使用较简单，但不支持融合操作。不同的是，BANG C自定义算子缺少类似CNNL这样的封装，算子的源码也在CATCH仓库中，所以要求用户有一定的BANG C开发知识, \*.mlu一类的脚本都是需要使用CNCC编译器去编译的。
     
     在 ``catch/torch_mlu/csrc/aten/operators/bang/internal/bang_internal.h`` 和 ``catch/torch_mlu/csrc/aten/operators/bang/internal/dump_internal.mlu`` 
     中分别添加kernel函数的声明和实现。

     ::

       bang_internal.h
       void dump(void *input, int32_t size, cnrtDim3_t dim, cnrtFunctionType_t ktype, cnrtQueue_t queue, cnrtDataType_t cnrt_type);

     ::
     
       dump_internal.mlu
       template <typename T>
       __mlu_func__ void dump_template(const char* format, T *input, int32_t size) {
           for(int i = 0; i < size; i++) {
             __bang_printf(format,*(input+i));
           }
       }
       
       __mlu_entry__ void dump_kernel(void *input, int32_t size, cnrtDataType_t cnrt_type) {
           if (cnrt_type == CNRT_FLOAT32) {
               dump_template<float>("%f\n", (float*)input, size);
           } else if (cnrt_type == CNRT_FLOAT16) {
               dump_template<half>("%hf\n", (half*)input, size);
           } else {
               __bang_printf("Invalid Data Type!!!");
           }
       }

       void dump(void *input, int32_t size, cnrtDim3_t dim, cnrtFunctionType_t ktype, cnrtQueue_t queue, cnrtDataType_t cnrt_type) {
           dump_kernel<<<dim, ktype, queue>>>(input, size, cnrt_type); // 启动 kernel
       }

   .. attention::
   
      | kernel一般以“cnnl_算子名_internal”命名。

5. 自定义正反向算子关联与实现。

   该步骤仅针对训练时使用的CNNL库或者BANG C实现的自定义算子，原生标准算子无需此步骤。正反向算子关联可以在C++侧或者Python侧完成，推荐用户在Python侧实现此步骤。具体实现
   与原生PyTorch一致，用户可以参考 ``pytorch/torch/autograd/function.py`` 文件。
   
   ::

     import torch
     from torch.autograd.function import Function

     class MLUDump(Function):
         @staticmethod
         def forward(ctx, a, b):
             # ctx 用于存储反向计算时的信息
             result = a + b

             # 调用正向自定义算子
             dump_success = torch.ops.torch_mlu.dump(result)

             # 保存反向计算需要使用的tensor
             ctx.save_for_backward(a, b, result)

             # 保存反向需要使用的非tensor数据
             ctx.dump_success = dump_success

             return result

         @staticmethod
         def backward(ctx, grad_output):
             # 返回的梯度个数应该与正向输入参数的个数一致
             # 不需要求梯度的参数或者非tensor的参数，其梯度返回None

             # 获取正向保存的tensor
             a, b, result = ctx.saved_tensors
             grad_input = grad_output + a + b * result
             if ctx.dump_success:
                 # 调用反向自定义算子
                 torch.ops.torch_mlu.dump(grad_input)
             # b不求梯度返回None
             return grad_input, None

     mlu_dump = MLUDump.apply
     a = torch.tensor([1.1, 2.2], requires_grad=True)
     b = torch.tensor([1.1, 2.2], requires_grad=False)
     out = mlu_dump(a.to('mlu'), b.to('mlu'))
     grad_output = torch.ones(out.shape, dtype=torch.float) * 2
     out.backward(grad_output.to('mlu'))

     """
     output:
     2.200000
     4.400000
     5.520000
     13.880000
     """

.. _自定义在线融合算子:

自定义在线融合算子
--------------------

.. _添加融合算子:

添加融合算子
~~~~~~~~~~~~~~~~~~~~

借助MagicMind后端可以提升PyTorch JIT模式下网络推理性能。根据MagicMind算子支持情况可将PyTorch IR图分割成若干个sub-graph，然后将这些sub-graph转成MagicMind中可进行优化和融合的Network来提升整网推理性能，具体工作流程如下图所示。

.. figure:: ../doc_image/mlu_magicmind_op.*

   MagicMind后端工作流程图

CATCH通过注册Pass的方式与PyTorch源码解耦，所有融合算子的实现都放在了CATCH中。融合算子可实现PyTorch IR Node到MagicMind Network Node的转换。

执行以下步骤在CATCH中添加MLU融合算子：

1. 添加kernel。

   Kernel中通过调用MagicMind库实现算子功能。以下以MagicMind库relu算子为例。

   算子的具体实现通过调用MagicMind接口来完成，具体如下所示：

   在 ``catch/torch_mlu/csrc/jit/codegen/convertion/ops/activation.cpp`` 中添加kernel函数的注册和实现。

   ::

     activation.cpp

     static auto registry = Registerer()
        .op("aten::relu(Tensor self) -> Tensor",
            [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
               torch::jit::Stack& params) -> bool {
              // 获取算子输入MagicMind ITensor指针
              auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);

              // 创建MagicMind Activation节点并添加到MagicMind Network中
              auto relu = handle->network->AddIActivationNode(input_tensor,
                                                              magicmind::IActivation::RELU);

              // 获取算子输出MagicMind ITensor指针
              auto output_tensor = relu->GetOutput(0);

              // 将torch::jit::Value与输出MagicMind ITensor绑定起来
              handle->bindingValueAndIvalue(
                  node->outputs()[0], codegen::bindITensor(output_tensor));
              return true;
            })
        .op("aten::relu_(Tensor(a!) self) -> Tensor(a!)",
            [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
               torch::jit::Stack& params) -> bool {
              // 获取算子输入MagicMind ITensor指针
              auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);

              // 创建MagicMind Activation节点并添加到MagicMind Network中
              auto relu = handle->network->AddIActivationNode(input_tensor,
                                                              magicmind::IActivation::RELU);

              // 获取算子输出MagicMind ITensor指针
              auto output_tensor = relu->GetOutput(0);

              // 将torch::jit::Value与输出MagicMind ITensor绑定起来
              handle->bindingValueAndIvalue(
                  node->outputs()[0], codegen::bindITensor(output_tensor));
              return true;
            });

   .. attention::
   
      | 同一类别的算子一般放在同一个.cpp文件中。

算子测试
~~~~~~~~~~~~~~~~~~~~
使用基于Python的unittest模块编写算子单元测试。测试时需提供相同的参数和输入数据，分别在MLU和CPU上执行算子，对比两者的输出结果。MLU和CPU计算结果因为量化等原因可能会产生差异，一般情况下两者的相对误差在以下范围内均是可以接受的：CNNL库和MagicMind单算子误差在0.3%以内（训练场景对算子精度要求较高）。

以下为代码示例：

- CNNL算子

  ::

    def test_div(self):
        shape_list =[(1, 2, 3, 4), (10, 10, 10, 10), (100, 200), (3, 40, 32), (1111), (99, 30, 40), (34, 56, 78, 90)]
        for shape in shape_list:
            x = torch.rand(shape, dtype = torch.float)
            y = torch.rand(shape, dtype = torch.float)
            y = y + 0.00005  # float range:[0.00005, 500]

            # test div(tensor, tensor)
            out_cpu = torch.div(x, y)
            out_mlu = torch.div(self.to_mlu(x), self.to_mlu(y))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True) # float type precision : 0.003

            out_cpu_1 = torch.div(x, 8)
            out_mlu_1 = torch.div(self.to_mlu(x), 8)

            # test div(scalar, scalar)
            while x.sum() > 400:
              x = x / 10
            out_cpu_2 = torch.div(x.sum(), 8.0)
            out_mlu_2 = torch.div(self.to_mlu(x).sum(), 8.0)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True) # float type precision : 0.003
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.003, use_MSE = True)  # float type precision : 0.003
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(), 0.003, use_MSE = True)  # float type precision : 0.003

- MagicMind算子

  ::

    def test_relu(self):
         for in_shape in [(1), (2, 3), (8, 224, 224), (1, 1, 1, 1), (1, 3, 16, 16),
                          (1, 3, 16, 16, 3)]:
             # 创建网络模型
             model = TestReluModel()

             # 创建输入Tensor
             input_x = torch.randn(in_shape)

             # Trace网络模型
             traced_model = torch.jit.trace(model, input_x, check_trace=False)

             # 创建MLU设备属性的输入Tensor
             input_x_mlu = input_x.to('mlu')

             # CPU前向推理
             out_cpu = model(input_x)
             # MLU前向推理
             out_mlu = traced_model(input_x_mlu)

             # CPU与MLU结果对比
             self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE = True)


内存管理
------------

Cambricon PyTorch支持对MLU内存的管理，通过对MLU上的内存管理提升性能。Cambricon PyTorch提供了一系列Python API，通过接口调用对正在使用/已经缓存的内存的检测以及释放缓存的操作。

调用以下接口返回当前正在使用的MLU内存，单位：字节。``device_index`` 默认值为-1，表示当前使用设备ID。

::

  torch_mlu.core.mlu_model.memory_allocated(int device_index)

调用以下接口返回当前缓存的MLU内存，其中包括使用中的内存和未使用的内存，单位：字节。``device_index`` 默认值为-1，表示当前使用设备ID。

::

  torch_mlu.core.mlu_model.memory_cached(int device_index)

调用以下接口返回在MLU上使用过的最大的MLU内存，单位：字节。``device_index`` 默认值为-1，表示当前使用设备ID。

::

  torch_mlu.core.mlu_model.max_memory_allocated(int device_index)


调用以下接口返回在MLU上缓存过的最大的MLU内存，单位：字节。``device_index`` 默认值为-1，表示当前使用设备ID。

::

  torch_mlu.core.mlu_model.max_memory_cached(int device_index)

调用以下接口清空所有MLU板卡上的所有缓存。``device_index`` 默认值为-1，表示当前使用设备ID。

::

  torch_mlu.core.mlu_model.empty_cached_memory(int device_index)

内存检查
~~~~~~~~~~~~

Cambricon PyTorch提供Debug Allocator功能检查当前框架是否存在非法MLU内存申请或者内存越界问题。

要开启Debug Allocator，只需将环境变量 ``ENABLE_CATCH_MEMORY_DEBUG`` 设置为1即可。程序会在内存释放的时候去检查这一段内存是否存在越界。

也可以使用Cambricon PyTorch提供的以下Python API来检查当前tensor是否申请合法以及当前申请的所有内存是否存在越界：

::
  
  torch_mlu.core.mlu_model.memory_debug(torch::Tensor tensor = None)

检查输入tensor申请是否合法，以及当前申请的所有内存是否存在越界。tensor默认值为None。

以下示例介绍如何调用该接口。调用之前需将环境变量 ``ENABLE_CATCH_MEMORY_DEBUG`` 设置为1。

.. code:: python
    
   import torch
   import torch_mlu.core.mlu_model as ct

   # 创建输入，并放上MLU
   x = torch.randn(30, 40, 10, 10, requires_grad=True).to(ct.mlu_device())
   x += 1
   ct.memory_debug(x)

如果不存在非法的内存申请或者内存越界，系统会打印如下结果。

::

  ===================== Checking Memory Out of Bound ...  =====================
  ===================== No Memory Out of Bound !!! =====================
  ===================== Storage is managed by allocator !!! =====================
  ===================== Checking Memory Out of Bound ...  =====================

如果检测到内存越界，Debug Allocator会打印出内存越界的内存块申请时候的调用栈，以帮助用户定位内存越界的位置。

::

  The memory is out of bound ! mask index = 0 ;
   origin header mask = 5497018662354700622 , now header mask = 5497018662354700622 ;
    origin footer mask = 5212716746423944008 , now footer mask = 41137136
    stack[0] : torch_mlu::MLUCachingAllocator::allocate(unsigned long, short) const
    stack[1] : torch_mlu::outBoundTest()
    stack[2] : torch_mlu::MLUAllocatorTest_allocate_Test::TestBody()
    stack[3] : void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*)
    stack[4] : testing::Test::Run()
    stack[5] : testing::TestInfo::Run()
    stack[6] : testing::TestCase::Run()
    stack[7] : testing::internal::UnitTestImpl::RunAllTests()
    stack[8] : testing::UnitTest::Run()

锁页内存
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cambricon PyTorch提供锁页内存对数据拷贝到设备上进行加速。Cambricon PyTorch分别支持对tensor和DataLoader锁页内存申请，接口和使用方式与原生PyTorch完全一致，当编译并导入CATCH之后就可以使用。

下面是一个对tensor使用锁页内存的用例：

.. code:: python

   #example:
   import torch
   import torch_mlu
   import torch_mlu.core.mlu_model as ct
    
   a = torch.randn([10,10]).float().pin_memory() # 拷贝到锁页内存
   a.to(ct.mlu_device(0)) # 拷贝到设备上
   
下面的用例展示了如何开启DataLoader的锁页功能并使用DataLoader导入数据：

.. code:: python

   import torch
   import torch_mlu
   from torch.utils.data import DataLoader
   import torch_mlu.core.mlu_model as ct
   
   train_loader = DataLoader(
                  train_dataset, # torch.utils.data.ImageFolder
                  batch_size=16,
                  shuffle=None,                                         
                  sampler=None,
                  num_workers=2, # 进程数
                  pin_memory=True) # 开启锁页内存
   
   for i, (images, target) in enumerate(train_loader):
       images = Variable(images.float(), requires_grad=False)
       images = images.to(ct.mlu_device())   

JIT图融合
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cambricon PyTorch提供接口控制是否使能JIT IR图融合机制，只有使能图融合机制才能使用MagicMind后端加速推理的过程，默认为使能状态。接口和使用方式与原生PyTorch类似，当编译并导入CATCH之后就可以使用。

以下为对图融合使能接口的用例：

.. code:: python

   #example:
   import torch
   import torch_mlu
   import torch_mlu.core.mlu_model as ct

   ct._jit_override_can_fuse_on_mlu(False) # 禁止图融合功能
