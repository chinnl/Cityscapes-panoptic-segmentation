import torch
import torch.nn as nn
import sys
# sys.path.append(r"E:\20231\DATN\detectron2_rebuild\Backbone")
from backbone_utils import *

class R50(nn.Module):
    def __init__(self) -> None:
        super(R50, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = (7,7), stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (3,3), stride = 2, padding = 1)
        )
        self.in_channels = 64
        self.layers = [3, 4, 6, 3] #Original config for R50 
        self.layer1 = self.make_layers(BottleneckBlock, self.layers[0], out_channels=64, stride = 1)
        self.layer2 = self.make_layers(BottleneckBlock, self.layers[1], out_channels=128, stride = 2)
        self.layer3 = self.make_layers(BottleneckBlock, self.layers[2], out_channels=256, stride = 2)
        self.layer4 = self.make_layers(BottleneckBlock, self.layers[3], out_channels=512, stride = 2)
        
    def forward(self, input: torch.Tensor):
        '''
        Input size: B x 3 x H x W
        '''
        x = self.stem(input)
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        
        output = [res5, res4, res3, res2]
        return output
    
    def make_layers(self, block, num_res_blocks, out_channels, stride):
        layers = []
        identity_downsample = None
        
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels*4)
            )
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_res_blocks-1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride=1):
        super(BottleneckBlock, self).__init__()
        self.expansion = 4
        self.bottle_neck_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False, padding = 0),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            nn.Conv2d(out_channels,out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0,  bias = False),
            nn.BatchNorm2d(out_channels*self.expansion),
        ) 
        self.identity_downsample = identity_downsample
    
    def forward(self, input):
        x = self.bottle_neck_conv(input)
        
        if self.identity_downsample is not None:
            short_cut = self.identity_downsample(input)

        else:
            short_cut = input
            
        return nn.ReLU()(x+short_cut)
    
if __name__ == '__main__':
    H = 800
    W = 600
    input = torch.zeros((5,3, H, W), dtype = torch.float)
    output = R50().forward(input)
    for op in output:
        print(op.shape)
