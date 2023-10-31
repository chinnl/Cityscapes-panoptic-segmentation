import torch
import torch.nn as nn
import sys
sys.path.append(r"E:\20231\DATN\detectron2_rebuild\Backbone")
from backbone_utils import *

class R50(nn.Module):
    def __init__(self) -> None:
        super(R50, self).__init__
        pass
    
    def forward(self, input: torch.Tensor):
        '''
        Input size: B x 3 x H x W
        '''
        stem_out = self.stem(input)
        
        res2 = self.res_block(stem_out, block_strides=[2]+[1]*2, short_cuts = [True]+[False]*2)
        
        res3 = self.res_block(res2, block_strides=[2]+[1]*3, short_cuts = [True]+[False]*3)
        
        res4 = self.res_block(res3, block_strides=[2]+[1]*5, short_cuts = [True]+[False]*5)
        
        res5 = self.res_block(res4, block_strides=[2]+[1]*2, short_cuts = [True]+[False]*2)
        
        output = [res5, res4, res3, res2]
        return output
    
    def stem(self, input):
        x = nn.Conv2d(3, 256, kernel_size = (7,7), stride = 2, padding = 3, bias = False)(input) 
        #padding = 3 for stem_output.shape = (B,256,H/2,W/2)
        x = nn.BatchNorm2d(256)(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size = (3,3), stride = 1, padding = 1)(x)
        return x
    
    def res_block(self, input, block_strides, short_cuts):
        x = input
        for stride, short_cut in zip(block_strides, short_cuts):
            x = BottleneckBlock(x, stride, short_cut)
        return x
    

if __name__ == '__main__':
    H = 800
    W = 600
    input = torch.zeros((5,3, H, W), dtype = torch.float)
    from time import time
    start = time()
    output = R50().forward(input)
    for op in output:
        print(op.shape)
    stop = time()
    print("Process time: ", stop - start)