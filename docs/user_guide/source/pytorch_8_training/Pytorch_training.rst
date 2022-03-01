Cambricon PyTorch（CATCH）支持使用MLU板卡进行单机单卡训练，同时也支持原生PyTorch分布式训练模块Distributed Data Parallel（DDP）进行单机多卡训练。

支持的典型网络
----------------
Cambricon PyTorch支持以下典型网络：

.. tabularcolumns:: |m{0.2\textwidth}|m{0.7\textwidth}|
.. table:: 支持的典型网络

   ========= ==================================================================
   类别      名称
   ========= ==================================================================
   分类网络  ResNet50、ResNet18、Inception v3、VGG16、VGG16 BN、Alexnet、MobileNet v2、ResNet101、VGG19、ResNet50 v1.5、Inception v2、DenseNet201、ShuffleNet V2 x0.5、Shufflenet V2 x1.0、Shufflenet V2 x1.5。

   检测网络  YoloV3、SSD-ResNet50、RetinaNet、SSD-VGG16、Mask R‑CNN‑ResNet101、Faster R‑CNN‑ResNet101。

   其他网络  Transformer、Fairseq、BERT、modelzoo-BERT、BERT-MSRA。
   ========= ==================================================================

.. attention::

   - 训练算子底层使用CNNL库。训练前要将其设置为CNNL。

   - 训练暂时未支持half模式训练。

异步执行
----------------
Cambricon PyTorch（CATCH）CNNL模式支持任务异步执行。

默认情况下，MLU操作为异步执行。当调用相关MLU算子时，这些任务会排队到对应设备上。主机不必等待MLU运行结束就可以进行其它计算。

在CPU和MLU之间拷贝数据时，会自动执行必要的同步。对于数据从CPU至MLU拷贝，拷贝操作 ``to()`` 和 ``copy_()`` 可以传入一个non_blocking参数以绕过同步。
异步拷贝可以有效并发拷贝任务和CPU侧程序运行，减小拷贝引入的时间消耗。

.. code:: python

   x = torch.randn(1000000, dtype=torch.float)
   x_mlu = x.to(torch.device('mlu'), non_blocking=True)

通过设置环境变量 ``CATCH_ASYNC_DISABLE=1`` 可强制进行同步计算。异步执行时只有在同步时才能返回任务运行状态，报错位置不一定是实际出错任务的位置， 也不会打印调试到错误调用栈。
因此在MLU操作异步运行发生错误时，可切换成同步运行进行调试。环境变量设定实际操作是会对每一个MLU算子任务进行同步，可以在每个MLU算子执行结束时返回报错信息。

.. code:: shell

   export CATCH_ASYNC_DISABLE=1


使用MLU设备
------------------------
Cambricon PyTorch （CATCH）允许设置当前使用的MLU卡，获取当前设备的MLU卡数量，并将数据放到指定的卡上。以下为代码示例：

.. code:: python

   import torch
   import torch_mlu
   import torch_mlu.core.mlu_model as ct

   #获取当前设备的卡数
   device_count = ct.device_count()
   #设置当前使用的设备的卡号
   ct.set_device(0)
   #将数据放到指定的卡上并进行计算
   input = torch.randn(8,3,224,224).to(ct.mlu_device())
   input1 = torch.abs(input)
   #将数据放到与输入tensor相同的卡上
   input2 = torch.randn(4,3,224,224).to(input1.device)

以上代码首先导入了与CATCH相关的torch_mlu库，使用 ``device_count()`` 获取当前设备的MLU卡数量；使用 ``set_device()`` 设置当前使用的卡号（卡号需小于当前设备的卡数）；使用 ``ct.mlu_device()`` 获取使用的设备；使用 ``to()`` 将数据放到指定的卡上，可以指定此前设置的卡，也可以指定与输入tensor相同的卡。

对于CPU与MLU之间的数据拷贝，或多个MLU设备间的数据拷贝。拷贝可以指定对应MLU设备以及设备号。或者通过Device Context Manager进行设备切换。

::

  CLASS torch.device

  可以通过设备类型指定确定需要的设备（‘cpu’或者‘mlu’），如果设备号没有指定，将会按照默认设备0设定。或者按照set_device()
  调用设定的当前设备。
  也可以在指定设备类型时传入需要使用的设备号，表示指定设备。

.. code:: python

   # tensor 通过to()拷贝指定设备`mlu:1`
   x = torch.randn((64, 3, 1080, 1920), dtype=torch.float)
   x_mlu = x.to(torch.device('mlu:1'))

::

  torch_mlu.core.mlu_model.Device(device)

  Device Context-manager 用于改变当前选择MLU设备
  参数：
      device（int) - 指定选择MLU设备号

.. code:: python

   # tensor 通过Device指定拷贝CPU至MLU 1卡
   y = torch.randn((64, 3, 1080, 1920), dtype=torch.float)
   with torch_mlu.core.mlu_model.Device(1):
       y_mlu = y.to(torch.device('mlu'))


.. _MLU未实现算子自动运行到CPU:

将MLU未实现算子自动运行到CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用环境变量 ``ENABLE_FALLBACK_TO_CPU`` 来控制打开和关闭该功能：

- 未定义该变量时，缺省为关闭自动运行到CPU功能。
- ``export ENABLE_FALLBACK_TO_CPU=0`` 时，关闭自动运行到CPU功能。
- ``export ENABLE_FALLBACK_TO_CPU=1`` 时，打开自动运行到CPU功能。

当 ``export ENABLE_FALLBACK_TO_CPU=1`` 时，Cambricon PyTorch训练过程中遇到MLU未实现的算子，会自动转移到CPU上运行，默认不打印算子的LOG信息。如果要打印在CPU上运行的算子信息，需配置环境变量 ``export TORCH_MIN_CNLOG_LEVEL=0`` 。

.. attention::

   - 只针对MLU未实现算子，不针对MLU上已注册的在执行过程中发生错误而转向CPU的算子。
   - 无法处理CPU不支持的算子或数据类型，例如，含有half类型的算子在转向CPU执行时会失败。
   - 不支持在第三方自定义的未在CPU上注册的算子。

.. _notifier如何使用:

使用notifier
-----------------------------
notifier是一种同步标记，用来统计执行时间、调整执行步调、同步不同的queue。

::

  CLASS torch_mlu.core.device.notifier.Notifier

以下是notifier提供的接口。

- place(Queue = None)

  将notifier标记放置在指定的Queue上，在不传入参数时，notifier将被放置在当前device对应的当前queue上。当前版本由于在device相同时，Python端创建出的Queue实际上总对应同一个当前queue，因而目前版本传入Queue参数效果与不传入相同，一般直接调用place接口，不传入参数。

- query()

  查询标记在queue上的notifier，之前的任务是否完成。返回布尔值。当为True时，表示任务已完成。

- synchronize()

  等待直到notifier标记之前的任务完成，使用该接口将阻塞调用的CPU线程。

- elapsed_time(end_notifier)

  返回start_notifier和end_notifier之间的总时间，单位为毫秒。传入的end_notifier需要与当前的notifier标记在同一个queue上。

- hardware_time(end_notifier)

  返回start_notifier和end_notifier之间的硬件时间，单位为微秒。传入的end_notifier需要与当前的notifier标记在同一个queue上。

- wait(Queue)

  该接口用于queue间同步，使得传入的queue等待该notifier完成。但当前版本Python端同一个设备仅有唯一的当前queue，因而该接口暂时无法使用。

.. attention::

   | notifier的place(Queue = None)和wait(Queue)接口要求notifier和queue在相同的device上。

.. code:: python

   import torch
   import torch_mlu.core.mlu_model as ct
   import torch_mlu.core.device.notifier as Notifier

   input1 = torch.randn(1000,3,2,2).to(ct.mlu_device())
   input2 = torch.randn(1000,3,2,2).to(ct.mlu_device())
   output = torch.neg(input1)
   #为计算时间，创建 start 和 end notifier
   start = Notifier.Notifier()
   end= Notifier.Notifier()
   #在开始计时处放置start notifier
   start.place()
   input1 = input3 * input2
   #在结束计时处放置notifier
   end.place()
   #同步notifier
   end.synchronize()
   #计算notifier之间的总时间
   time = start.elapsed_time(end)

上述代码展示了使用notifier计时的过程：

1. 创建两个notifier实例，然后在需要计时的代码之前以及结束后，分别调用相应notifier的place接口，放置notifier。

2. 调用end notifier的synchronize接口，确保end标记之前的任务完成（也可以根据需要使用Queue的synchronize接口）。

3. 计算start和end notifier之间的硬件时间。

在此过程中也可调用 ``query()`` 接口查看notifier状态。

单机单卡训练
------------------------
单机单卡训练的整体流程主要包括数据加载、模型初始化与量化转换、前向传播、反向传播、优化器更新权重等。关于单机单卡训练流程的更多信息，可访问 ``catch/examples/training/single_card_demo.py`` 。可使用Python直接运行该demo。

分布式训练
------------------------
Cambricon PyTorch （CATCH）目前支持对tensor进行单机多卡和多机多卡的卡间集合通信，对网络进行单机多卡和多机多卡的分布式训练。

**Tensor的卡间集合通信**

目前支持以下通信原语：卡间广播（broadcast）、卡间规约（allreduce、reduce）、卡间收集（allgather），支持进程间同步（barrier）。其中，卡间规约又包含4种操作：求和、求连乘、求最大值、求最小值。Demo程序展示了通信操作的主要步骤：启动子进程、创建进程组实例、调用通信接口、销毁进程组实例。

**Tensor的卡间点对点通信**

目前支持以下通信原语：发送（send）和接收（recv）。

.. attention::

   - 当前点对点通信需要用户保障调用顺序合理，源进程中的send操作需对应目标进程中的recv操作。
     
   - 当前不支持在一个设备内部进行点对点通信。
     
   - 当前需要用户设置环境变量使能点对点通信功能（export CNCL_SEND_RECV_ENABLE=1）。


**网络的多卡训练**

原生框架的Distributed Data Parallel分布式训练机制在单个机器上有两种使用模式：a、单进程多卡；b、多进程多卡，每个进程用一张卡（官方推荐模式）。

MLU现在仅支持多进程多卡。通过在每次前向计算前对param/buffer作broadcast同步，等反向计算出梯度后，再对梯度做allreduce规约（用多个进程上计算出的梯度的均值替换各自进程计算出的梯度）以完成不同进程上的训练过程之间的通信。Demo程序展示了多卡训练的主要步骤：启动子进程、创建进程组实例、初始化数据预处理、模型训练、销毁进程组实例。多机多卡功能相比于单机多卡功能，在使用上需要注意在创建进程组实例时，传入的IP为rank 0卡所在机器的IP地址，其他用法跟单卡一致。

分布式相关功能Demo路径： ``catch/examples/training/multi_card_demo.py、catch/examples/training/multi_card_demo_deprecated.py`` 。

.. attention::

   | 目前在MLU设备上使用原生框架DDP多进程多卡模式分布式相关功能时，需注意以下内容：

   - 如果对64位数据类型数据进行通信，因为当前MLU最大只支持32位数据类型，int64和float64只可用于对应的32位数据类型所表示范围内数据。

   - reduce和allreduce通信原语当前只支持浮点数据类型、int8和uint8类型。

   - 当前不支持原生SyncBatchNorm（分布式场景下的BatchNorm算子）功能。

   - 当前，如果采用MLU-Link或者PCIe互联技术进行通信，存在超时时间这一限制，默认为1小时。如果单卡的通信任务执行时间超出该时间，将会终止该通信任务。

   - 当前不支持一个进程内的网络模型参数分布在多张MLU卡上。

   - 当前不建议在非默认的MLU Queue上运行DDP。

     以下为示例代码：

     .. code:: python

        with torch_mlu.core.mlu_model.Queue(): # 创建一个新queue
            model = DDP(...)
            loss = model(input.to('mlu')).sum())
            loss.backward()
            grad = model.module.weight.grad()
            average = dist.all_reduce(grad)  # average和grad可能会不相等
        ...


.. _模型的存储与加载:

模型的存储与加载
-----------------------------

Cambricon PyTorch支持原生的模型存储与加载方法。

.. code:: python

   torch.save({
              'epoch':epoch,
              'model_state_dict':model.state_dict(),
              'optimizer_state_dict':optimizer.state_dict(),
              'loss':loss,
              ...
              }, PATH)
   ...
   model = TheModelClass(*args, **kwargs)
   optimizer = TheOptimizerClass(*args, **kwargs)
   ...
   checkpoint = torch.load(PATH)
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']
   ...
   model.eval() or model.train()
   ...


以下为使用示例。

.. code:: python

   #从原生模型的checkpoint中加载模型
   def test_load_from_cpu():
       net = models.resnet18()
       net = net.to(ct.mlu_device())
       optimizer = torch.optim.SGD(net.parameters(), 0,
                                   momentum=0.9,
                                   weight_decay=1e-4)
       checkpoint = torch.load('net_cpu.pth')
       net.load_state_dict(checkpoint['state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer'])
       return net

.. 模型推理:

模型推理
-------------------------
Cambricon PyTorch训练完毕之后可以在MLU上以及在GPU上进行浮点推理。

以下为使用示例。

.. code:: python

   #使用GPU进行浮点推理
   resume_point = torch.load(args.resume, map_location=torch.device('cpu'))
   model.load_state_dict(resume_point, strict=False)
   model.to(torch.device("cuda"))
   model.eval()
   ...

   #使用MLU进行浮点推理
   resume_point = torch.load(args.resume, map_location=torch.device('cpu'))
   model.load_state_dict(resume_point, strict=False)
   model.to("mlu")
   model.eval()
   ...

任务调度
------------

为同时满足MLU端串行程序运行和并行程序运行，寒武纪提供了Queue功能，可以将计算任务或内存拷贝任务下发到特定的Queue运行。

Queue的核心思想如下：

1. 任务下发到Queue后异步执行。

2. 同一个Queue内的任务按下发先后顺序串行执行。

3. 不同Queue间的任务并行执行。

可以将需要串行执行的任务下发至同一个Queue，将需要并行执行的任务下发到不同的Queue。单个Queue内的任务将按照创建顺序执行，不同的Queue会按照相对顺序并发执行。

通常情况下，无需创建新的Queue。默认情况下，每个设备使用其自己的默认的Queue。如果没有指定设备，将会使用当前使用设备对应的Queue。

你可以调用接口获取目前使用的Queue进行操作，例如获取当前使用Queue对象显示调用同步功能synchronize。具体接口使用介绍，参见 :ref:`Python API` 。


.. _Python API:

Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  CLASS torch_mlu.core.device.Queue

包含以下内容

- **device_index(int)**

  Queue对应的设备。device_index默认值为-1， 表示使用当前设备。

- **synchronize()**

  对当前Queue所有内核任务进行同步。

- **query()**

  查询Queue中任务完成状态。返回True时，表示Queue中执行的任务全部完成；返回False时，表示Queue中执行的任务正在运行。

::

  torch_mlu.core.mlu_model.current_queue(int device_index)

调用该接口获取指定设备的当前Queue ID。其中，device_index默认值为-1，表示当前设备ID。

::

  torch_mlu.core.mlu_model.default_queue(int deivce_index)


调用该接口获取指定设备默认Queue。其中，device_index默认值为-1，表示当前设备ID。

::

  torch_mlu.core.mlu_model.synchronize(int device_index)

调用该接口对指定设备的Queue所有内核任务进行同步。其中，device_index默认值为-1，表示当前使用设备ID。

::

  torch_mlu.core.mlu_model.Queue(device)

调用该接口用于改变当前使用的Queue，作用域中所有MLU算子操作将会排列在当前设定的Queue上。其中，``device`` 用来指定MLU设备号，为int类型。



基本用法
============

以下为代码示例：

.. code:: python

   import torch
   import torch_mlu
   import torch_mlu.core.mlu_model as ct
   ct.set_device(0)

   # 当前设备0卡使用默认Queue计算
   x = torch.randn((64, 128, 24, 24), dtype=torch.float32)
   x_mlu = x.to(torch.device('mlu'))
   out = torch.abs(x_mlu)
   ...

   # 设定MLU 1卡的Queue进行计算
   with torch_mlu.core.mlu_model.Queue(1):
       x = torch.randn((64, 128, 24, 24), dtype=torch.float32)
       x_mlu = x.to(torch.device('mlu'))
       out = torch.abs(x_mlu)
   ...

   # 设定MLU 2卡的Queue进行计算
   with torch_mlu.core.mlu_model.Queue(2):
       x = torch.randn((64, 128, 24, 24), dtype=torch.float32)
       x_mlu = x.to(torch.device('mlu'))
       out = torch.abs(x_mlu)
   ...


性能分析工具Profiler
------------------------------------------

Profiler是PyTorch中自带的性能分析工具，用于统计算子时间，分析性能瓶颈，进行有针对性的性能优化。在原生Profiler基础上，寒武纪针对MLU硬件特点，有效扩展了Profiler的功能，使其在MLU设备上统计CPU/MLU算子硬件计算时间；查看算子、网络调用层次；统计网络中算子调用、整体硬件时间等情况。

.. code:: python

   import torch
   import torch_mlu.core.mlu_model as ct


   def test_profiler(self):
       # 创建输入，并放上MLU
       x = torch.randn(30, 40, 10, 10, requires_grad=True).to(ct.mlu_device())
       # 根据输入形状构建全1Tensor，并放上MLU
       grad = torch.ones(x.shape).to(ct.mlu_device())

       # 进入Profile环境
       with torch.autograd.profiler.profile(use_mlu=True) as prof:
           y = x * 2 + 4
           y.backward(grad)

       print(prof)

上述示例将数据放上MLU，并在Profiler中显式使用 ``use_mlu=True`` ，由此进入Profiler环境。以下为性能分析结果。

.. figure:: ../doc_image/profiler_1.*

   性能分析结果

其中，第一列是算子的名称，第2-5列分别为算子的CPU时间占总时间比例、CPU花费时间，第6-11列分别为CNNL算子的MLU时间占总时间的比例、算子硬件计算时间、同一算子的平均调用时间、算子调用次数，以及算子处理的tensor的形状。

从第2行开始为代码中先后调用的算子，跑完前向后，继续跑反向求梯度。有个异常情况是，MulBackward0算子调用了mul的算子实现，两个算子属于前者调用后者的关系，所以两者的硬件计算时间相同，同时在计算总时间时，这个时间被计算了两次，相应的百分比因此产生了一些变化，这个属于Profiler的原生设计，无法修改，只是在查看算子时间时，需要了解这一点情况。

使用Profiler统计的MLU硬件计算时间与使用CNRT工具统计的硬件时间基本一致，最大误差约为1 μs。由于运行程序时不是独占服务器运行，不同时刻统计到的硬件时间会有差异，最大误差约为1 μs，即在运行同一程序时，同一算子花费的硬件时间误差在1 μs内，基本可以认为Profiler工具工作正常。

此外，Profiler工具还提供以下功能和选项：

- **record_shapes**

  如果需查看算子处理的tensor形状，可以在 ``torch.autograd.profiler.profile(use_mlu=True)`` 中添加属性参数 ``record_shapes=True`` ，则打印出的信息会增加shape信息。

  .. figure:: ../doc_image/profiler_2.*

     record_shapes打开后的效果图

  使用该参数时，内部处理会增加保存形状等操作，因此得到的CPU时间会有增加，MLU时间基本无变化。

- **export_chrome_trace(path)**

  输出一个EventList对象，作为Chrome追踪工作的输入文件。
  执行 ``prof.export_chrome_trace("./chrom_trace")``
  在本地得到一个chrom_trace的文件，然后在谷歌浏览器中输入 ``Chrome:://Tracing``，将该文件拖入，即得到如下时间开销图：

  .. figure:: ../doc_image/chrom_json.*

     chrom_trace加载JSON文件结果

  在左上方的 ``Process CPU functions`` 中可以看到在CPU上运行的算子的调用情况。点击相应算子，可以显示在CPU上的计算时间，在 ``Process MLU functions`` 中可以看到3个小块，对应三个在MLU上计算的算子。

  将加载的JSON图按W键放大，可以得到如下放大图，可以看到有三个算子放在了MLU上计算，点击相应算子，可以在屏幕左下方显示硬件计算时间。

  .. figure:: ../doc_image/chrom_json_2.*

     chrom_trace加载JSON文件放大图

- **profile_memory**

  bool类型，默认关闭，开启时表示追踪tensor内存的分配与释放。

- **table显示**

  使用table显示打印的prof信息，并按照指定列名进行排序。

  将 ``print(prof)`` 改成 ``print(prof.table(sort_by="self_cpu_time_total"，row_limit=10, header="TEST"))`` ，则对prof数据按照 ``self_cpu_time_total`` 进行排序，设定 ``row_limit`` 为10单位，表格名称设为 ``TEST``。

  .. figure:: ../doc_image/profiler_4.*

     table效果图

- **key_averages()**

  对所有函数输出其平均时间。

  以下为效果图：

  .. figure:: ../doc_image/profiler_5.*

      key_averages效果图

- **total_average()**

  计算所有时间的平均值。

  ``<FunctionEventAvg key=Total self_cpu_time=7.011ms cpu_time=994.866μs mlu_time=17.900μs input_shapes=None>``
  ``self_cpu_time`` 是 ``key_averages()`` 表格中 ``Self CPU total`` 列之和， ``cpu_time`` 是 ``key_averages()`` 表格中 ``Self CPU total`` 列的总和除以总的 ``Number of Calls`` 次数，``mlu_time`` 是 ``MLU total`` 列的总和除以 ``Number of Calls`` 总和，得到的平均值。

- **添加label**

  在Python代码块中增加一个标签，方便后续进行代码追踪。示例如下：

  .. code:: python

     def test_record_function(self):
         x = torch.randn(10, 10).to(ct.mlu_device())

         with profile(use_mlu=True) as p:
             x = x + 1.5
             with record_function("label"):
                 y = x * 2 + 4

  结果如下图所示。

  .. figure:: ../doc_image/profiler_6.*

     label效果图

  从结果中可以看到，label标签在add算子与mul算子之间，它相当于把 ``y = x * 2 + 4`` 这行代码整合成一个代码块，用 ``label`` 进行标记，同时它的MLU硬件时间是mul算子与add算子的MLU硬件时间之和，方便将这行代码作为一个整体进行处理。

- **overhead**

  使用profiler时，不可避免地会引入overhead。在开启profiler前后，测试ResNet50，E2E时间会增加0.87%，而MLU硬件时间增加0.05%。E2E时间overhead较大是因为在CPU上执行的函数中插入了回调函数，影响了速度；而抓取硬件时间是由底层依赖软件优化过的，开销较小。总体来看，profiler产生的overhead在合理范围内。

.. _performancebestpractice:

性能最佳实践
------------------------
Cambricon PyTorch （CATCH）提供以下性能优化选项和工具。

.. _ioqueue:

IO Queue
~~~~~~~~~~~~~~~~~~~~~~~~

目前支持使用IO Queue实现数据集从主机侧（host）到设备侧（device）的拷贝与设备计算并行执行的功能。

使用环境变量 ``USE_IO_QUEUE`` 来打开和关闭该功能，默认为关闭。

- ``export USE_IO_QUEUE=OFF`` 时，关闭IO Queue功能。
- ``export USE_IO_QUEUE=ON`` 时，打开IO Queue功能。

当 ``USE_IO_QUEUE=ON`` 时，Cambricon PyTorch训练过程中使用的 ``torch.utils.data.DataLoader`` 会自动将当前batch及下一个batch的数据集通过IO Queue从主机拷贝至设备。

.. attention::

   - 开启该功能后，仅在使能 ``pin_memory`` 且 ``num_workers`` 大于0的 ``torch.utils.data.DataLoader`` 中生效。

   - 开启该功能后，会预读取一个batch的数据集，这将产生额外的内存开销，进而导致部分网络在相同设备上的最大可运行batch size减小。

   - 开启该功能后，``torch.utils.data.DataLoader`` 返回的数据集已拷贝到设备，与原生返回数据集在主机侧的行为可能不一致。对于部分依赖原生行为的网络，需要修改网络实现才可正常运行。





