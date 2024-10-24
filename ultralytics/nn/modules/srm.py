import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

__all__ = (
    "SRM_conv2d",
)


cwd = os.getcwd()  # Get the current working directory (cwd)
SRM_npy = np.load(cwd+'/ultralytics/nn/modules/SRM_Kernels.npy')

class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1,1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, \
                        self.stride, self.padding, self.dilation, \
                        self.groups)