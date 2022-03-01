# coding=utf-8
import warnings
import torch
from torch.nn.parameter import Parameter

class MLUModel(object):
    def __init__(self):
        self.weight = None
        self.bias = None
        self.scale = None
        self.quantized_mode = None
        self.input_mean = None
        self.input_std = None
        self.running_mean = None
        self.running_var = None
        self.num_batches_tracked = None

    def set_bias_with_zeros(self, out_channels):
        self.bias = Parameter(torch.zeros(out_channels))

    def set_weight_with_ones(self, out_channels):
        self.weight = Parameter(torch.ones(out_channels))

    def set_weight_bias(self, w, b):
        if isinstance(w, Parameter):
            self.weight = w
            self.bias = b
        else:
            self.weight = Parameter(w, requires_grad = False)
            if b is not None:
                self.bias = Parameter(b, requires_grad = False)
            else:
                self.bias = None

    def set_weight(self, w):
        if isinstance(w, Parameter):
            self.weight = w
        else:
            self.weight = Parameter(w, requires_grad = False)

    def set_scale(self, s, scales_channel = None):
        if scales_channel is not None:
            s = torch.cat((s, scales_channel)).float()
        self.scale = Parameter(s, requires_grad = False)

    def set_quantized_mode(self, q, q_channel = None):
        if not isinstance(q, torch.Tensor):
            warnings.warn("The quantized_mode is not a tensor, please use Tensor type!")
            q = torch.tensor(q, dtype=torch.int)
        if q_channel and q.item() == 1:
            quantized_mode = torch.tensor([3], dtype=torch.int)
        elif q_channel and q.item() == 2:
            quantized_mode = torch.tensor([4], dtype=torch.int)
        else:
            quantized_mode = q.int()
        self.quantized_mode = Parameter(quantized_mode, requires_grad = False)

    def set_mean_std(self, mean, std):
        if mean.dim() == 1:
            self.input_mean = Parameter(mean.reshape(1, -1, 1, 1), requires_grad=False)
            self.input_std = Parameter(std.reshape(1, -1, 1, 1), requires_grad=False)
        else:
            self.input_mean = Parameter(mean, requires_grad=False)
            self.input_std = Parameter(std, requires_grad=False)

    def set_running_mean(self, running_mean):
        self.running_mean = running_mean

    def set_running_var(self, running_var):
        self.running_var = running_var

    def set_num_batches_tracked(self, num_batches_tracked):
        self.num_batches_tracked = num_batches_tracked

    def set_running_mean_with_zeros(self, num_features):
        self.running_mean = Parameter(torch.zeros(num_features))

    def set_running_var_with_ones(self, num_features):
        self.running_var = Parameter(torch.ones(num_features))

    def set_num_batches_tracked_with_zero(self):
        self.num_batches_tracked = Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
