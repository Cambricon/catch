from __future__ import print_function

import sys
import os
import copy
# import time
import unittest
import logging
# import numpy as np

import torch
import torch.nn as nn
# from torch.nn import Parameter
# import torch.nn.functional as F
# import torch_mlu
import torch_mlu.core.mlu_model as ct

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)

# Because the calculation errors of Linear may introduce the relu backward sign
# to change and introduce greater errors threshold, set the input random seed
# to ensure that the data is determinted.
torch.manual_seed(6503)

linear1 = nn.Linear(128, 100)
linear2 = nn.Linear(100, 128)

class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.

        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        #mean = x.relu()
        mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        # output = self.gamma * (x - mean) / (std + self.epsilon) + self.beta
        return mean


class feedforward(nn.Module):

    def __init__(self, in_channels, num_units=None):
        '''Point-wise feed forward net.

        Args:
          in_channels: a number of channels of inputs
          num_units: A list of two integers.
        '''
        if num_units is None:
            num_units = [100, 128]
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units

        # nn.Linear is faster than nn.Conv1d
        self.conv1 = nn.Sequential(
            nn.Linear(self.in_channels, self.num_units[0]), nn.ReLU())
        self.conv1[0].weight = linear1.weight
        self.conv1[0].bias = linear1.bias
        self.conv2 = nn.Linear(self.num_units[0], self.num_units[1])
        self.conv2.weight = linear2.weight
        self.conv2.bias = linear2.bias
        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        # Residual connection
        outputs += inputs
        # outputs = outputs + inputs

        outputs = self.normalization(outputs)
        return outputs


class TestOps(TestCase):

    # @unittest.skip("not test")
    @testinfo()
    def test_linear(self):
        for _ in range(1):

            class Net_cpu(nn.Module):
                def __init__(self):
                    super(Net_cpu, self).__init__()
                    self.f = feedforward(128)

                def forward(self, x):
                    output = self.f(x)
                    return output

            class Net_mlu(nn.Module):
                def __init__(self):
                    super(Net_mlu, self).__init__()
                    self.f = feedforward(128)

                def forward(self, x):
                    output = self.f(x)
                    return output

            model_cpu = Net_cpu()
            model_cpu.train().float()
            model_mlu = Net_mlu()

            model_mlu.train().float().to(ct.mlu_device())
            x = torch.randn((32, 36, 128), dtype=torch.float, requires_grad=True)
            x_mlu = copy.deepcopy(x)
            grad = torch.ones((32, 36, 1), dtype=torch.float)
            grad_mlu = copy.deepcopy(grad)
            self.assertTensorsEqual(x, x_mlu, 0.0, use_MSE=True)
            out_cpu = model_cpu(x)
            out_mlu = model_mlu(self.to_mlu(x_mlu))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.02, use_MSE=True)
            out_cpu.backward(grad)
            out_mlu.backward(self.to_mlu(grad_mlu))
            self.assertTensorsEqual(x.grad, x_mlu.grad, 0.03, use_MSE=True)

            self.assertTensorsEqual(model_cpu.f.conv1[0].bias.grad,
                                    model_mlu.f.conv1[0].bias.grad.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(model_cpu.f.conv1[0].weight.grad,
                                    model_mlu.f.conv1[0].weight.grad.cpu(), 0.01, use_MSE=True)
            self.assertTensorsEqual(model_cpu.f.conv2.bias.grad,
                                    model_mlu.f.conv2.bias.grad.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(model_cpu.f.conv2.weight.grad,
                                    model_mlu.f.conv2.weight.grad.cpu(), 0.01, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
