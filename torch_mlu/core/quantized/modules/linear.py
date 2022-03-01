from __future__ import division
import sys
import math
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear as NNLinear
from torch.nn.parameter import Parameter
import torch.nn.init as init
# torch_mlu MUST be imported, which is a namespace
from .mlumodel import MLUModel

class MLULinear(torch.nn.Module, MLUModel):
    __constants__ = ["bias", 'in_features', 'out_features', 'scale', 'quantized_mode']
    def __init__(self, in_features, out_features, bias=True):
        super(MLULinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # By default, we define scale as torch.nn.Parameter
        self.scale = None
        self.quantized_mode = None
        self.quant = False
        # TODO: To support asymmetric quantization later.
        self.use_symmetry = Parameter(torch.tensor([1]), requires_grad = False)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if (input.device.type != 'mlu'):
            # run quantization in cpu
            return torch.nn.functional.linear(input, self.weight, self.bias)
        else:
            # run linear with scale in mlu
            return torch.ops.torch_mlu.linear(input,
                                              self.weight,
                                              self.bias,
                                              self.scale,
                                              self.quantized_mode)

    def _observer_forward_hook(self, cls, input, output):
        return self.observer(*input)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(MLULinear, self)._save_to_state_dict(destination, prefix, keep_vars)
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

            if (self.observer.dtype == 'int8'):
                self.set_quantized_mode(torch.Tensor([1]))
            elif (self.observer.dtype == 'int16'):
                self.set_quantized_mode(torch.Tensor([2]))
            else:
                self.set_quantized_mode(torch.Tensor([1]))

            # retrieve tensor from parameter
            destination[prefix+'quantized_mode'] = self.quantized_mode.data
            if 'MLUMinMaxObserver' == weight_observer._get_name():
                self.set_scale(torch.tensor([input_observer.scale,weight_observer.scale]))
            else:
                destination[prefix+'quantized_by_channel'] = torch.tensor([1]).data
                destination[prefix+'scale_weight_per_channel'] = torch.tensor(weight_observer.scales).data
                self.set_scale(torch.tensor([input_observer.scale, 0.0]))
            destination[prefix+'scale'] = self.scale.data

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(MLULinear, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                     missing_keys, unexpected_keys, error_msgs)
        key_s = prefix + 'scale'
        key_q_m = prefix + 'quantized_mode'
        key_s_w = prefix + 'scale_weight_per_channel'
        key_q_c = prefix + 'quantized_by_channel'

        if key_s in state_dict.keys():
            self.set_scale(state_dict[key_s], state_dict.get(key_s_w, None))
        if key_q_m in state_dict.keys():
            self.set_quantized_mode(state_dict[key_q_m], state_dict.get(key_q_c, None))

        if prefix + 'bias' not in state_dict.keys() or state_dict[prefix + 'bias'] is None:
            self.set_bias_with_zeros(self.out_features)

    def _get_name(self):
        return 'MLULinear'

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, scale={}, quantized_mode={}, quant={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.scale, self.quantized_mode,
            self.quant)

    @classmethod
    def from_float(cls, mod, gen_quant):
        assert type(mod) == NNLinear, 'MLULinear only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'

        qlinear = MLULinear(mod.in_features, mod.out_features)
        qlinear.set_weight_bias(mod.weight, mod.bias)
        if gen_quant == True:
            qlinear.quant = True
            qlinear.qconfig = mod.qconfig
            # find and set observer
            # instance *.qconfig.activation()
            qlinear.observer = qlinear.qconfig.activation()
            # instance hook func
            qlinear.register_forward_hook(qlinear._observer_forward_hook)
        return qlinear
