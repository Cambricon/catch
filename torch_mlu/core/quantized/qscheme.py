'''
Default quantization scheme
The options are:
    iteration: Specify numbers of images for quantization
    use_avg: Whether to use AVG method to calculate scale
    data_scale: Specify data_scale to scale Min and Max
    mean: When firstconv need mean preprocess input
    std: When fistconv need std preprocess input
    per_channel: Whether to quantize weight for each channel
'''
mlu_qscheme = {
    'iteration' : 1,
    'use_avg'   : False,
    'data_scale': 1.0,
    'mean'      : [0,0,0],
    'std'       : [1,1,1],
    'firstconv' : True,
    'per_channel': False
}

CONV_HYPER_PARAMS = {
    'alpha' : 0.04,
    'beta' : 0.1,
    'gamma' : 2,
    'delta' : 100,
    'th' : 0.03,
}

LINEAR_HYPER_PARAMS = {
    'alpha' : 0.04,
    'beta' : 0.1,
    'gamma' : 2,
    'delta' : 100,
    'th' : 0.01,
}

class QuantifyParams(object):
    r"""Base class for all quantify module params."""

    def __init__(self):
        self.init_bitwidth = 8
        self.max_bitwidth = 31
        self.quantify_rate = 0.01
        self.alpha = 0.04
        self.beta = 0.1
        self.gamma = 2
        self.delta = 100
        self.th = 0.03
        self.steps_per_epoch = 10000
 
    def set_bitwidth(self,
                     init_bitwidth = 8,
                     max_bitwidth = 31):
        self.init_bitwidth = init_bitwidth
        self.max_bitwidth = max_bitwidth
 
    def set_hyperparam(self,
                       quantify_rate = 0.01,
                       alpha = 0.04,
                       beta = 0.1,
                       gamma = 2,
                       delta = 100,
                       th = 0.03):
        self.quantify_rate = quantify_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.th = th
 
class MatmulQuantifyParams(QuantifyParams):
 
    def __init__(self):
        super(QuantifyParams, self).__init__()
        self.set_hyperparam(th = 0.01)
 
class ConvQuantifyParams(QuantifyParams):
 
    def __init__(self):
        super(QuantifyParams, self).__init__()
        self.set_hyperparam(th = 0.03)
