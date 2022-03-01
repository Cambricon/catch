from torch import nn
# cnq is short for Cambricon neuware quantized
import torch_mlu.core.quantized.modules as cnq

# Mapping for swapping float module to quantized ones
DEFAULT_MLU_MODULE_MAPPING = {
    nn.Linear: cnq.MLULinear,
    nn.Conv3d: cnq.MLUConv3d,
    nn.Conv2d: cnq.MLUConv2d,
    nn.Conv1d: cnq.MLUConv1d,
    'Conv2d' : cnq.MLUConv2d,
}

DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST = (
    nn.Linear,
    nn.Conv3d,
    nn.Conv2d,
    nn.Conv1d,
    'Conv2d',
)
