import torch
from torch import nn, Tensor
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, flag_bias, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=flag_bias, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, True, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, False, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, False, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, False, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, False, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, False, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, True, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class linear(nn.Module):
    def __init__(self, in_num, out_num, bias_flag):
        super(linear, self).__init__()
        self.features = nn.Linear(in_num, out_num, bias=bias_flag)

    def forward(self, x):
        output = self.features(x)
        return output

class two_linear(nn.Module):
    def __init__(self):
        super(two_linear, self).__init__()
        self.lin1 = linear(224, 64, True)

        self.lin2 = linear(64, 16, False)

    def forward(self, x):
        lin1 = self.lin1(x)
        outputs = self.lin2(lin1)
        return outputs


input = torch.randn(1,64,224,224,dtype=torch.float)
model = InceptionA(64,64)
model = torch.jit.trace(model, input)
model_script = torch.jit.script(InceptionA(64,64))
model.save("../build/bin/cnnl/incepA.pt")
model_script.save("../build/bin/cnnl/incepA_script.pt")

input1 = torch.randn(64, 224,dtype=torch.float)
model1 = two_linear()
model1 = torch.jit.trace(model1, input1)
model_script1 = torch.jit.script(two_linear())
model1.save("../build/bin/cnnl/linear.pt")
model_script1.save("../build/bin/cnnl/linear_script.pt")
