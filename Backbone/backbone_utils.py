import torch.nn as nn
import torch

def BottleneckBlock(input, stride, with_shortcut = False):
    input_dim = input.shape[1]
    x = nn.Conv2d(in_channels = input_dim, out_channels = input_dim//2, kernel_size = 1, stride = stride, bias = False)(input)
    x = nn.BatchNorm2d(input_dim//2)(x)
    x = nn.ReLU()(x)
    x = nn.Conv2d(in_channels = input_dim//2, out_channels = input_dim//2, kernel_size = 3, padding = 1, bias = False)(x)
    x = nn.BatchNorm2d(input_dim//2)(x)
    x = nn.ReLU()(x)
    x = nn.Conv2d(in_channels = input_dim//2, out_channels = input_dim, kernel_size = 1)(x)
    x = nn.BatchNorm2d(input_dim)(x)
    if with_shortcut:
        short_cut = nn.Conv2d(in_channels = input_dim, out_channels = input_dim, kernel_size = 1, stride = stride, bias = False)(input)
        short_cut = nn.BatchNorm2d(input_dim)(short_cut)
        return nn.ReLU()(x+short_cut)
    else: 
        return nn.ReLU()(x)

    