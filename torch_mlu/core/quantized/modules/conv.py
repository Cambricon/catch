# coding=utf-8
from __future__ import division
import math
import logging
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.modules.conv import Conv3d as NNConv3d
from torch.nn.modules.conv import Conv2d as NNConv2d
from torch.nn.modules.conv import Conv1d as NNConv1d
from torch.nn.parameter import Parameter
from .mlumodel import MLUModel

import torch.nn.init as init
from typing import Tuple, List, Dict, Optional
# torch_mlu MUST be imported, which is a namespace
import torch_mlu

class MLUConvNd(torch.nn.Module, MLUModel):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size',
                     'scale', 'quantized_mode', 'input_std', 'input_mean']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(MLUConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels    must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if self.transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # mlu quantize
        self.scale = None
        self.quantized_mode = None
        # TODO: To support asymmetric quantization later.
        self.use_symmetry = Parameter(torch.tensor([1]), requires_grad = False)
        # mlu first_conv
        self.input_std = None
        self.input_mean = None
        self.quant = False
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _observer_forward_hook(self, cls, input, output):
        return self.observer(*input)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(MLUConvNd, self)._save_to_state_dict(destination, prefix, keep_vars)
        if self.weight is not None:
            destination[prefix+'weight'] = self.weight
        if self.bias is not None:
            destination[prefix+'bias'] = self.bias

        if self.quant == True:
            # 1. calculate the scale of weight
            weight_observer = self.qconfig.weight()
            weight_observer(self.weight)
            weight_observer.calculate_qparams()

            # 2. calculate the scale of input
            input_observer = self.observer
            input_observer.calculate_qparams()
            if self.observer.dtype == 'int8':
                self.set_quantized_mode(torch.Tensor([1]))
            elif self.observer.dtype == 'int16':
                self.set_quantized_mode(torch.Tensor([2]))
            else:
                self.set_quantized_mode(torch.Tensor([1]))

            if 'MLUPerChannelMinMaxObserver' == weight_observer._get_name():
                destination[prefix+'quantized_by_channel'] = torch.tensor([1])
                destination[prefix+'scale_weight_per_channel'] = torch.tensor(weight_observer.scales)
                self.set_scale(torch.tensor([input_observer.scale, 0.0]))
            else:
                self.set_scale(torch.tensor([input_observer.scale, weight_observer.scale]))

            destination[prefix+'scale'] = self.scale.data
            destination[prefix+'quantized_mode'] = self.quantized_mode.data

            if self.input_std is not None:
                destination[prefix+'input_mean'] = self.input_mean.data.mul(255)
                destination[prefix+'input_std'] = self.input_std.data.mul(255)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(MLUConvNd, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                     missing_keys, unexpected_keys, error_msgs)
        key_s = prefix + 'scale'
        key_q_m = prefix + 'quantized_mode'
        key_s_w = prefix + 'scale_weight_per_channel'
        key_q_c = prefix + 'quantized_by_channel'
        if key_s in state_dict.keys():
            self.set_scale(state_dict[key_s], state_dict.get(key_s_w, None))
        if key_q_m in state_dict.keys():
            self.set_quantized_mode(state_dict[key_q_m], state_dict.get(key_q_c, None))
        if prefix+'input_mean' in state_dict.keys() and prefix+'input_std' in state_dict.keys():
            self.set_mean_std(state_dict[prefix+'input_mean'], state_dict[prefix+'input_std'])

        if prefix + 'bias' not in state_dict.keys() or state_dict[prefix + 'bias'] is None:
            self.set_bias_with_zeros(self.out_channels)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__) + \
               ', scale={}, quantized_mode={}, input_mean={}, input_std={}' \
               .format(self.scale, self.quantized_mode, self.input_mean, self.input_std)

class MLUConv2d(MLUConvNd):
    # judge whether this layer is the first conv layer
    # only the first conv layer needs to use firstconv OP
    is_first_layer = True
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MLUConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return torch.nn.functional.conv2d(
                       torch.nn.functional.pad(input, expanded_padding, mode='circular'),
                       weight, self.bias, self.stride,
                       _pair(0), self.dilation, self.groups)
        return torch.nn.functional.conv2d(input, weight, self.bias, self.stride,
                                          self.padding, self.dilation, self.groups)

    def forward(self, input):
        if input.device.type != 'mlu':
            # run CPU mode
            return self.conv2d_forward(input, self.weight)
        else:
            # run MLU mode
            if self.input_std is None:
                # run conv without first
                return torch.ops.torch_mlu.conv2d(input, self.weight, self.bias,
                    list(self.padding), list(self.stride), list(self.dilation),
                    self.groups, self.scale, self.quantized_mode)
            else:
                # run conv with first
                norm_input = (input - self.input_mean) / self.input_std
                return torch.ops.torch_mlu.conv2d(norm_input, self.weight,
                    self.bias, list(self.padding), list(self.stride), list(self.dilation),
                    self.groups, self.scale, self.quantized_mode)
    def _get_name(self):
        return 'MLUConv2d'

    @classmethod
    def from_float(cls, mod, gen_quant):
        assert type(mod) == NNConv2d or issubclass(type(mod), NNConv2d), \
               'MLUConv2d only works for nn.Conv2d or its subclass'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'

        qconv2d = MLUConv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                            mod.stride, mod.padding, mod.dilation,
                            mod.groups, mod.bias is not None, mod.padding_mode)
        qconv2d.set_weight_bias(mod.weight, mod.bias)

        if gen_quant == True:
            qconv2d.quant = True
            if MLUConv2d.is_first_layer:
                # set first conv
                configuration = mod.qconfig.activation().qscheme
                if (configuration['firstconv']) is True and \
                    (configuration['mean'] is None or
                    configuration['std'] is None):
                    assert False, \
                        """
                        \rYou are using firstconv but forget setting mean or std,
                        \rplease set mean and std in qconfig!
                        """
                elif (configuration['firstconv']) is True and \
                    configuration['mean'] is not None and \
                    configuration['std'] is not None:
                    assert qconv2d.in_channels in [1, 3, 4], \
                        """
                        \rYou are using firstconv for the 1st MLUConv2d layer,
                        \rmake sure your input channel for the 1st MLUConv2d layer is 1(GRAY) or 3(RGB) or 4(RGBA),
                        \rand now your input channel is {}.
                        \rIf you want to use generic conv, please set firstconv False in qconfig!
                        """.format(qconv2d.in_channels)
                    assert min(configuration['mean']) >= 0 and max(configuration['mean']) <= 1 and \
                        min(configuration['std']) >= 0 and max(configuration['std']) <= 1, \
                        """
                        \rYou are using firstconv for the 1st MLUConv2d layer,
                        \rmake sure your mean and std set in qconfig between [0.0, 1.0],
                        \rwe will multiply 255 inside Pytorch for the real inference process.
                        \rNow mean is: {}, std is: {}.
                        \rIf you want to use generic conv, please set firstconv False in qconfig!
                        """.format(configuration['mean'], configuration['std'])

                    if list(set(configuration['mean'])) == [0] and list(set(configuration['std'])) == [1]:
                        logging.warning(
                            """
                            \rYou are using firstconv, default mean [0, 0, 0] std [1, 1, 1] are used,
                            \rplease check whether default mean and std value is corresponding to
                            \ryour expectation. If not, set mean and std manually.
                            \rIf you want to use generic conv, please set firstconv False in qconfig!""")
                    qconv2d.set_mean_std(
                        torch.Tensor(configuration['mean']), torch.Tensor(configuration['std']))
                MLUConv2d.is_first_layer = False

            qconv2d.qconfig = mod.qconfig
            # find and set observer
            # instance *.qconfig.activation()
            qconv2d.observer = qconv2d.qconfig.activation()
            # instance hook func
            qconv2d.register_forward_hook(qconv2d._observer_forward_hook)
        return qconv2d


class MLUConv1d(MLUConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MLUConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv1d_forward(self, input, weight):
        return torch.nn.functional.conv1d(input, weight, self.bias, self.stride,
                                          self.padding, self.dilation, self.groups)

    def forward(self, input):
        if input.device.type != 'mlu':
            # run CPU mode
            return self.conv1d_forward(input, self.weight)
        else:
            # run MLU mode
            return torch.ops.torch_mlu.conv1d(input,
                                              self.weight,
                                              self.bias,
                                              list(self.padding),
                                              list(self.stride),
                                              list(self.dilation),
                                              self.groups,
                                              self.scale,
                                              self.quantized_mode)
    def _get_name(self):
        return 'MLUConv1d'

    @classmethod
    def from_float(cls, mod, gen_quant):
        assert type(mod) == NNConv1d, 'MLUConv1d only works for nn.Conv1d'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'

        qConv1d = MLUConv1d(mod.in_channels, mod.out_channels, mod.kernel_size,
                            mod.stride, mod.padding, mod.dilation,
                            mod.groups, mod.bias is not None, mod.padding_mode)

        qConv1d.set_weight_bias(mod.weight, mod.bias)
        if gen_quant == True:
            qConv1d.quant = True
            configuration = mod.qconfig.activation().qscheme
            qConv1d.qconfig = mod.qconfig
            # find and set observer
            # instance *.qconfig.activation()
            qConv1d.observer = qConv1d.qconfig.activation()
            # instance hook func
            qConv1d.register_forward_hook(qConv1d._observer_forward_hook)
        return qConv1d


class MLUConv3d(MLUConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(MLUConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

    def conv3d_forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return torch.nn.functional.conv3d(torch.nn.functional.pad(input, expanded_padding, mode='circular'),
                                              self.weight, self.bias, self.stride, _triple(0),
                                              self.dilation, self.groups)
        return torch.nn.functional.conv3d(input, self.weight, self.bias, self.stride,
                                          self.padding, self.dilation, self.groups)
    def forward(self, input):
        if input.device.type != 'mlu':
            # run CPU mode
            return self.conv3d_forward(input)
        else:
            return torch.ops.torch_mlu.conv3d(input,
                                              self.weight,
                                              self.bias,
                                              list(self.padding),
                                              list(self.stride),
                                              list(self.dilation),
                                              self.groups,
                                              self.scale,
                                              self.quantized_mode)

    def _get_name(self):
        return 'MLUConv3d'

    @classmethod
    def from_float(cls, mod, gen_quant):
        assert type(mod) == NNConv3d, 'MLUConv3d only works for nn.Conv3d'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'

        qConv3d = MLUConv3d(mod.in_channels, mod.out_channels, mod.kernel_size,
                            mod.stride, mod.padding, mod.dilation,
                            mod.groups, mod.bias is not None, mod.padding_mode)

        qConv3d.set_weight_bias(mod.weight, mod.bias)
        if gen_quant == True:
            qConv3d.quant = True
            configuration = mod.qconfig.activation().qscheme
            assert configuration['per_channel'] == False, \
                """
                \rPlease check your qconfig, MLUConv3d do not support
                \rper_channel quantization, set per_channel False in qconfig!
                """
            qConv3d.qconfig = mod.qconfig
            # find and set observer
            qConv3d.observer = qConv3d.qconfig.activation()
            # instance hook func
            qConv3d.register_forward_hook(qConv3d._observer_forward_hook)
        return qConv3d
