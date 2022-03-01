'''
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import copy
import os
import torch
import torch.nn as nn
import itertools
import torch_mlu.core.mlu_model as ct
from .quantized.default_mappings import (DEFAULT_MLU_MODULE_MAPPING,
                                         DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST)
from .quantized.qconfig import QConfig, default_mlu_qconfig, custom_mlu_qconfig, norm_mlu_qconfig, mlu_activation_qconfig, mlu_per_channel_qconfig
from .quantized.observer import *

from .quantized.qscheme import MatmulQuantifyParams, ConvQuantifyParams

def compose_qconfig(weight_observer, activation_observer):
    return QConfig(weight = weight_observer, activation = activation_observer)

def swap_module(mod, gen_quant, mapping):
    '''
        mod: Is a Op like 'nn.Linear'
        mapping': Is a Dict for replacing Op with q_Op, like {nn.Linear:cnq:MLULinear}
    '''
    new_mod = mod
    # Always replace dequantstub with dequantize
    if type(mod) in mapping:
        new_mod = mapping[type(mod)].from_float(mod, gen_quant)
    elif mod._get_name() in mapping:
        new_mod = mapping[mod._get_name()].from_float(mod, gen_quant)
    return new_mod

def convert(module, gen_quant, mapping=None, inplace=False):
    reassign = {}
    SWAPPABLE_MODULES = ()

    for name, mod in module.named_children():
        if type(mod) not in SWAPPABLE_MODULES:
            convert(mod, gen_quant, mapping, inplace=True)
        reassign[name] = swap_module(mod, gen_quant, mapping)

    for key, value in reassign.items():
        module._modules[key] = value

def _observer_forward_hook(self, input, output):
    '''
        define a hook function for hook module's internal values
        self: layer
        input: input tensor
        output: output tensor
    '''
    return self.observer(output)

def add_observer_(module):
    '''
    Args:
        module: module with qconfig
    return:
        module is modified inplace with added observer modules and forward_hooks
    '''
    # recursive call add_observer_
    # Magic happens here!!!
    # if child is 'Linear', it will be attached with observer qconfig.activation
    for child in module.children():
        if hasattr(child, 'qconfig') and child.qconfig is not None:
            child.observer = child.qconfig.activation()
        else:
            add_observer_(child)

    if hasattr(module, 'qconfig') and module.qconfig is not None and \
        len(module._modules) == 0:
        # observer and hook will be gone after we swap the module
        module.add_module('observer', module.qconfig.activation())
        module.register_forward_hook(_observer_forward_hook)

def _propagate_qconfig_helper(module, qconfig_tuple, mixed_layer=None,
                              white_list=None, qconfig_parent=None,
                              prefix=''):
    '''
    Args:
        white_list: list of quantizable modules
    return:
        module is modified inplace with qconfig attached
    '''
    if white_list is None:
        white_list = DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST

    qconfig_spec = qconfig_tuple[1]
    if mixed_layer is not None:
        for keys, dtype in mixed_layer.items():
            if prefix in [key.strip(" ") for key in keys.split(",")]:
                if dtype in {'float16', 'int8', 'int16'}:
                    qconfig_spec = {
                        nn.Linear : custom_mlu_qconfig(dtype, qconfig_tuple[0]),
                        nn.Conv3d : custom_mlu_qconfig(dtype, qconfig_tuple[0]),
                        nn.Conv2d : custom_mlu_qconfig(dtype, qconfig_tuple[0]),
                        nn.Conv1d : custom_mlu_qconfig(dtype, qconfig_tuple[0]),
                        'Conv2dStaticSamePadding' : custom_mlu_qconfig(dtype, qconfig_tuple[0]),
                        'Conv2d' : custom_mlu_qconfig(dtype, qconfig_tuple[0]),
                    }


    module_qconfig = qconfig_spec.get(type(module), qconfig_parent)
    module_qconfig = qconfig_spec.get(prefix, module_qconfig)
    # To get qconfig by module name. This is convenient for end users when handling
    # with customed operator
    module_qconfig = qconfig_spec.get(module._get_name(), module_qconfig)
    module_qconfig = getattr(module, 'qconfig', module_qconfig)

    if type(module) in white_list:
        module.qconfig = module_qconfig
    if module._get_name() in white_list:
        module.qconfig = module_qconfig

    for name,child in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        _propagate_qconfig_helper(child, qconfig_tuple, mixed_layer,
                                  white_list, module_qconfig,
                                  module_prefix)

def propagate_qconfig_(module, qconfig_tuple=None, mixed_layer=None):
    '''
    assign 'qconfig' attribute on each leaf module
    '''
    if qconfig_tuple is None:
        qconfig_tuple = {}
    _propagate_qconfig_helper(module, qconfig_tuple, mixed_layer)

def quantize_dynamic_mlu(model, qconfig_spec=None, dtype=None,
                         mapping=None, inplace=False, gen_quant=False, mixed_layer=None):

    # step0: print mixed_quantization layers
    if mixed_layer is not None:
        if isinstance(mixed_layer, dict):
            keys = list(mixed_layer.keys())
            data_type = list(mixed_layer.values())[0].strip(" ")
            print("The layers : ", keys, " will be quantized by ", data_type)
        else:
            raise ValueError("type(mixed_layer) must be dict, now is ", type(mixed_layer))
    qconfig_spec_list = [qconfig_spec]

    # step1: create a map dict according to qconfig_spec
    if qconfig_spec is None:
        # DEFAULT qconfig WITHOUT gen quantization
        qconfig_spec = {
            nn.Linear : default_mlu_qconfig,
            nn.Conv3d : default_mlu_qconfig,
            nn.Conv2d : default_mlu_qconfig,
            nn.Conv1d : default_mlu_qconfig,
            'Conv2d' : default_mlu_qconfig,
            }
    elif isinstance(qconfig_spec, dict):
        if gen_quant:
            if dtype in {'int8', 'int16'}:
                weight_observer = mlu_weight_observer(dtype)
                if qconfig_spec.get('per_channel', False):
                    weight_observer = mlu_weight_per_channel_observer(dtype, qconfig_spec)
                activation_observer = mlu_activation_observer_dict[qconfig_spec.get('method', 'minmax')](dtype, qconfig_spec)

                qconfig_spec = {
                        nn.Linear : compose_qconfig(weight_observer, activation_observer),
                        # plugin Conv3d does not support per channel
                        nn.Conv3d : compose_qconfig(mlu_weight_observer(dtype), activation_observer),
                        nn.Conv2d : compose_qconfig(weight_observer, activation_observer),
                        nn.Conv1d : compose_qconfig(weight_observer, activation_observer),
                        'Conv2d' : compose_qconfig(weight_observer, activation_observer),
                        }
            else:
                raise ValueError("Unknow dtype for quantization")
        else:
            qconfig_spec = {
                    nn.Linear : custom_mlu_qconfig(dtype, qconfig_spec),
                    nn.Conv3d : custom_mlu_qconfig(dtype, qconfig_spec),
                    nn.Conv2d : custom_mlu_qconfig(dtype, qconfig_spec),
                    nn.Conv1d : custom_mlu_qconfig(dtype, qconfig_spec),
                    'Conv2d' : custom_mlu_qconfig(dtype, qconfig_spec),
                    }
    else:
        raise ValueError("Unknown qconfig_spec")
    qconfig_spec_list.append(qconfig_spec)

    # step2: assign mapping value
    if mapping is None:
        mapping = DEFAULT_MLU_MODULE_MAPPING

    # step3: deep copy model for quantization
    if not inplace:
        try:
            model = copy.deepcopy(model)
        except:
            inplace = True
            print("Warning: Failed to deepcopy model, \"inplace\" is set to True by default.")

    model.eval()
    # step4: go through such model
    propagate_qconfig_(model, qconfig_spec_list, mixed_layer)

    # step5: add observer_
    if gen_quant is True:
        add_observer_(model)

    # step6: convert model to qmodel
    convert(model, gen_quant, mapping, inplace=True)
    return model