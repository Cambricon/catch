MLU使用基础
----------------------------
Cambricon PyTorch主要使用场景为模型推理和训练。训练场景更注重可扩展性，典型的加速手段是数据并行和模型并行，
由于要处理节点间任务调度、通讯和同步，通常要把网络拆解成细粒度的算子，无法做到端到端执行。

针对上述情形，寒武纪CNNL算子库提供了手工优化的基本算子或简单的融合算子，保证每个算子的单次运行延时尽可能低。

推理场景更注重运行速度，通过原生JIT模式进行融合模式的推理。寒武纪MagicMind库提供了图融合，图优化等机制，保证融合模式下的推理效率更高。

基础使用
""""""""""""""""""""""""""""""

导入torch_mlu相关包
=================================

::

  import torch_mlu
  import torch_mlu.core.mlu_quantize as mlu_quantize
  import torchvision.models as models

.. _加载模型:

加载模型
=================================
Cambricon PyTorch模型在Torchvision库中均已提供。使用时，通过Torchvision导入模型，具体命令如下：

::

  torch.set_grad_enabled(False) #注意：在运行MagicMind推理融合模式时，必须设置该条件。
  net = getattr(models, 'net_name')(pretrained=True)

``models`` 为Torchvision导入后的模型；``getattr(models,net_name)`` 将返回指定的网络对象；``pretrained=True`` 将在初始化对象过程中同时加载权重。

.. _推理快速入门:

推理快速入门
----------------------------
PyTorch使用由多层互连计算单元组成的神经网络（模型）进行模型推理。PyTorch提供了设计好的模块和类便于快速创建网络模型，例如 ``torch.nn`` 类。本节使用 ``torch.nn`` 类来定义MNIST网络模型并介绍模型推理的具体方法。
完整代码可以参见本节末尾的 推理完整代码_。

编译和安装
"""""""""""""""""""""""""""
使用模型推理前，要先编译与安装Cambricon PyTorch，并进入PyTorch的虚拟环境。更多信息，参见 :ref:`编译和安装` 章节。

导入必要模块
"""""""""""""""""""""""""""
执行以下命令导入模块：
::

  import torch
  import torch_mlu
  import torch.nn as nn
  import torch.nn.functional as F
  torch.set_grad_enabled(False)

定义和初始化模型
"""""""""""""""""""""""""""
示例模型使用PyTorch的卷积全连接等运算串联为简单网络。

定义模型的 ``Net`` 类，需执行以下步骤：

1. 编写引用nn.Module的__init__函数。

   在__init__中定义连接在网络中的所有层。这里将遵循标准MINST算法使用卷积创建输入图像通道为1，输出10个标签目标的网络模型，这些标签代表数字0到9。

   ::
   
     class Net(nn.Module):
       def __init__(self):
         super(Net, self).__init__()
   
         self.conv1 = nn.Conv2d(1, 32, 3, 1)
         self.conv2 = nn.Conv2d(32, 64, 3, 1)
   
         self.dropout1 = nn.Dropout2d(0.25)
         self.dropout2 = nn.Dropout2d(0.5)
   
         self.fc1 = nn.Linear(9216, 128)
         self.fc2 = nn.Linear(128, 10)
   
     my_nn = Net()
     print(my_nn)

2. 编写forward函数。该函数会将输入数据传递到网络的计算图，完成模型的前向推理过程。

   ::
   
     class Net(nn.Module):
         def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(1, 32, 3, 1)
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.dropout1 = nn.Dropout2d(0.25)
           self.dropout2 = nn.Dropout2d(0.5)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)
     
         def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
     
           x = self.conv2(x)
           x = F.relu(x)
     
           x = F.max_pool2d(x, 2)
           x = self.dropout1(x)
           x = torch.flatten(x, 1)
           x = self.fc1(x)
           x = F.relu(x)
           x = self.dropout2(x)
           x = self.fc2(x)

           output = F.log_softmax(x, dim=1)
           return output

3. 将上述模型定义保存为test_network.py。

使用JIT对模型进行trace
"""""""""""""""""""""""""""

::

  n = Net().eval().float()
  example_forward_input = torch.rand((1,1,28,28), dtype=torch.float)

  module = torch.jit.trace(n, example_forward_input)

模型推理
"""""""""""""""""""""""""""

::

  input_data=torch.randn((1,1,28,28))
  input_mlu = input_data.to("mlu")
  module.to("mlu")
  output=module(input_mlu)
  print(output.cpu())

运行 ``python test_network.py`` 进行模型推理。

由于权重和量化都是随机数生成的，本结果仅作为示例参考。

以下为模型推理完整代码：

.. _推理完整代码:

::

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch_mlu
  torch.set_grad_enabled(False)
  
  class Net(nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
      def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
  
  n = Net().eval().float()
  example_forward_input = torch.rand((1,1,28,28), dtype=torch.float)

  module = torch.jit.trace(n, example_forward_input)
  
  input_data=torch.randn((1,1,28,28))
  input_mlu = input_data.to("mlu")
  module.to("mlu")
  output = module(input_mlu)
  print(output.cpu())


训练快速入门
---------------------------
本节以MNIST为例说明具体的训练流程。更多详细信息，参见本节末尾的 训练完整代码_。

编译和安装
"""""""""""""""""""""""""""
使用模型训练前，要先编译与安装Cambricon PyTorch，并进入PyTorch的虚拟环境。更多信息，参见 :ref:`编译和安装` 章节。

导入必要模块
"""""""""""""""""""""""""""
执行以下命令导入模块：

::

  import torch
  import numpy as np
  from torch.utils.data import DataLoader
  from torchvision.datasets import mnist
  from torch import nn
  from torch import optim
  from torchvision import transforms
  import torch.nn.functional as F
  
  import torch_mlu.core.mlu_model as ct           

其中，``ct`` 模块用于管理MLU设备，将数据、模型在CPU与MLU间进行拷贝，并对MLU设备、MLU Queue等模块进行管理。

定义和初始化模型
"""""""""""""""""""""""""""

MINIST示例模型使用PyTorch构建模块，从输入图像中提取某些特征（如边缘检测、清晰度、模糊度）进行图像识别。

定义模型的 ``Net`` 类。需执行以下步骤：

1. 编写一个继承nn.Module的__init__函数。在__init__函数中定义了连接在网络中的基本层。

   ::
   
     class Net(nn.Module):
         def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(1, 32, 3, 1)
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.dropout1 = nn.Dropout2d(0.25)
           self.dropout2 = nn.Dropout2d(0.5)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)

2. 编写forward函数。该函数会将输入数据传递到网络的计算图中并完成模型前向计算。

   ::
   
     def forward(self, x):
       x = self.conv1(x)
       x = F.relu(x)
       x = self.conv2(x)
       x = F.relu(x)
       x = F.max_pool2d(x, 2)
       x = self.dropout1(x)
       x = torch.flatten(x, 1)
       x = self.fc1(x)
       x = F.relu(x)
       x = self.dropout2(x)
       x = self.fc2(x)
       output = F.log_softmax(x, dim=1)
       return output

3. 将上述模型定义保存为minist.py。

准备数据集以及数据预处理模块
""""""""""""""""""""""""""""""""""""""""""""
::

  data_tf = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize([0.1307],[0.3081])])
  train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
  test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
  train_data = DataLoader(train_set, batch_size=64, shuffle=True)
  test_data = DataLoader(test_set, batch_size=64, shuffle=False)

此处使用PyTorch中的预处理操作，``data_tf`` 将输入数据转化为Tensor、并进行Normalize预处理，然后将数据按 ``batch_size`` 大小加载。

为保证训练效果，在训练集上使用 ``shuffle=True`` 将训练图片打乱，但是在测试集上不打乱图片。

训练集与测试集均直接从网上下载，并自动进行预处理，因此必须保证机器联网。

如果无法访问互联网资源，可将相应的数据提前拷贝至 ``mnist.py`` 所在目录，按 ``./data/raw/****`` 方式存放，并将上面的 train_set/test_set设置如下：

::

  train_set = mnist.MNIST('./data', train=True)
  test_set = mnist.MNIST('./data', train=False)

定义训练与验证模块
"""""""""""""""""""""""""""
将以下代码加入到 ``mnist.py`` 以构建训练模块和验证模块。

::

  nums_epoch = 10  # 此处设置的10个训练epoch
  save_model = True  # 保存模型开关
  
  def train(model, train_data, optimizer, epoch):
    model = model.train()
    for batch_idx, (img, label) in enumerate(train_data):
        img = img.to(ct.mlu_device())
        label = label.to(ct.mlu_device())
        optimizer.zero_grad()
        out = model(img)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_data.dataset),
                100. * batch_idx / len(train_data), loss.item()))

  def validate(val_loader, model):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(ct.mlu_device())
            target = target.to(ct.mlu_device())
            output = model(images)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred.cpu()
            target = target.cpu()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

其中，``nums_epoch`` 为训练的epoch数，用来控制训练的轮数。此处只作为演示用途，用户根据实际情况设置该值。``save_model`` 用来设置是否保存模型，默认值为True。

模型训练
"""""""""""""""""""""""""""

执行 ``python mnist.py``，得到如下结果（此处只显示第0个epoch的训练和验证数据）：

::

  Train Epoch: 0 [0/60000      (0%)]    Loss: 2.291011
  Train Epoch: 0 [6400/60000  (11%)]    Loss: 0.513938
  Train Epoch: 0 [12800/60000 (21%)]	Loss: 0.485264
  Train Epoch: 0 [19200/60000 (32%)]	Loss: 0.259880
  Train Epoch: 0 [25600/60000 (43%)]	Loss: 0.246993
  Train Epoch: 0 [32000/60000 (53%)]	Loss: 0.273036
  Train Epoch: 0 [38400/60000 (64%)]	Loss: 0.095428
  Train Epoch: 0 [44800/60000 (75%)]	Loss: 0.102112
  Train Epoch: 0 [51200/60000 (85%)]	Loss: 0.161822
  Train Epoch: 0 [57600/60000 (96%)]	Loss: 0.354688

  Test set: Average loss: 0.0650, Accuracy: 9812/10000 (98%)

保存模型
"""""""""""""""""""""""""""
在验证接口后，可以将训练好的模型保存到指定位置，此处命名为 ``model.pth``。

::

    if epoch == nums_epoch - 1:
        checkpoint = {"state_dict":net.state_dict(), "optimizer":optimizer.state_dict(), "epoch": epoch}
        torch.save(checkpoint, 'model.pth')

当 ``epoch`` 达到 ``nums_epoch-1`` 时，会在本地保存模型，名为 ``model.pth``，该模型是直接保存的MLU模型。

以下为模型训练完整代码：

.. _训练完整代码:

::

  import torch
  import os
  import numpy as np
  from torch.utils.data import DataLoader
  from torchvision.datasets import mnist
  from torch import nn
  from torch import optim
  from torchvision import transforms
  from torch.optim.lr_scheduler import StepLR
  
  import torch.nn.functional as F
  import torch_mlu.core.mlu_model as ct
  
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 32, 3, 1)
          self.conv2 = nn.Conv2d(32, 64, 3, 1)
          self.dropout1 = nn.Dropout2d(0.25)
          self.dropout2 = nn.Dropout2d(0.5)
          self.fc1 = nn.Linear(9216, 128)
          self.fc2 = nn.Linear(128, 10)
  
      def forward(self, x):
          x = self.conv1(x)
          x = F.relu(x)
          x = self.conv2(x)
          x = F.relu(x)
          x = F.max_pool2d(x, 2)
          x = self.dropout1(x)
          x = torch.flatten(x, 1)
          x = self.fc1(x)
          x = F.relu(x)
          x = self.dropout2(x)
          x = self.fc2(x)
          output = F.log_softmax(x, dim=1)
          return output
  
  def train(model, train_data, optimizer, epoch):
      model = model.train()
      for batch_idx, (img, label) in enumerate(train_data):
          img = img.to(ct.mlu_device())
          label = label.to(ct.mlu_device())
          optimizer.zero_grad()
          out = model(img)
          loss = F.nll_loss(out, label)
          loss.backward()
          optimizer.step()
          if batch_idx % 100 == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(img), len(train_data.dataset),
                  100. * batch_idx / len(train_data), loss.item()))
  
  def validate(val_loader, model):
      test_loss = 0
      correct = 0
      model.eval()
      with torch.no_grad():
          for images, target in val_loader:
              images = images.to(ct.mlu_device())
              target = target.to(ct.mlu_device())
              output = model(images)
              test_loss += F.nll_loss(output, target, reduction='sum').item()
              pred = output.argmax(dim=1, keepdim=True)
              pred = pred.cpu()
              target = target.cpu()
              correct += pred.eq(target.view_as(pred)).sum().item()
      test_loss /= len(val_loader.dataset)
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(val_loader.dataset),
          100. * correct / len(val_loader.dataset)))
  
  def main():
      os.environ["TORCH_MIN_CNLOG_LEVEL"] = "3"
      data_tf = transforms.Compose(
                  [transforms.ToTensor(),
                   transforms.Normalize([0.1307],[0.3081])])
   
      train_set = mnist.MNIST('./data',train=True,transform=data_tf,download=True)
      test_set = mnist.MNIST('./data',train=False,transform=data_tf,download=True)
       
      train_data = DataLoader(train_set,batch_size=64,shuffle=True)
      test_data = DataLoader(test_set,batch_size=1000,shuffle=False)
       
      net_orig = Net()
      net = net_orig.to(ct.mlu_device())
      optimizer_orig = optim.Adadelta(net_orig.parameters(), 1)
      optimizer = ct.to(optimizer_orig, torch.device("mlu"))
       
      nums_epoch = 10
      save_model = True
  
      scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
      for epoch in range(nums_epoch):
          train(net, train_data, optimizer, epoch)
          validate(test_data, net)
  
          scheduler.step()
          if save_model:
              if epoch == nums_epoch-1:
                  checkpoint = {"state_dict":net.state_dict(), "optimizer":optimizer.state_dict(), "epoch": epoch}
                  torch.save(checkpoint, 'model.pth')
          
  if __name__ == '__main__':
      main()
