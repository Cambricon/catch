import os
import torch
from torch import nn
import torch_mlu.core.mlu_model as ct
os.environ['TORCH_MIN_CNLOG_LEVEL'] = '-1'

# cnnl test
x = torch.randn((1,2,3,3), dtype=torch.float, requires_grad=True)
y = torch.randn((1,2,3,3), dtype=torch.float, requires_grad=True)
x_mlu = x.to(ct.mlu_device())
y_mlu = y.to(ct.mlu_device())
res = x_mlu + y_mlu

model = nn.AvgPool2d(2).train().float()
model.to(ct.mlu_device())
res = model(x_mlu)
grad_mlu = torch.randn((res.shape), dtype=torch.float).to(ct.mlu_device())
res.backward(grad_mlu)

res = torch.clamp(x_mlu, min=-1.0, max=1.0)
res = torch.cat([x_mlu, y_mlu], dim=0)

x = torch.randn((1,3,112,112), dtype=torch.float, requires_grad=True)
x_mlu = x.to(ct.mlu_device())
model = nn.Conv2d(3, 16, kernel_size=3).to(ct.mlu_device())
res = model(x_mlu)
model = nn.Conv2d(3, 16, kernel_size=3, bias=False).to(ct.mlu_device())
res = model(x_mlu)
