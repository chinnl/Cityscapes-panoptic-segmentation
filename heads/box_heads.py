import numpy as np
from typing import List
import torch
from torch.nn import Conv2d, ReLU, Flatten, Linear
import torch.nn as nn
from layers import ShapeSpec, get_norm

class Fast_RCNN_Conv_FC_Head(nn.Sequential):
    def __init__(self,
                 input_shape: ShapeSpec,
                 *,
                 conv_dims: List[int],
                 fc_dims: List[int],
                 conv_norm = ""):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.conv_norm_relus = []
        
        for k, conv_dim in enumerate(conv_dims):
            conv = nn.Sequential(
                Conv2d(self._output_size[0],
                       conv_dim,
                       kernel_size=3,
                       padding=1,
                       bias=not conv_norm),
                get_norm(conv_norm, conv_dim),
                ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        
        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", Flatten())
            fc = Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim
        
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x