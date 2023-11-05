import torch
import torch.nn as nn


C = 256
class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        self.backbone = backbone
        
    def forward(self, input: torch.Tensor):
        bottom_ups = self.backbone.forward(input) #[res5, res4, res3, res2]
        top_downs = []
        for idx, P_i in enumerate(bottom_ups):
            lat_connect = nn.Conv2d(C, C, kernel_size = 1, stride = 1, bias = False)(P_i)
            lat_connect = nn.BatchNorm2d(C)(lat_connect)
            if len(top_downs)>0:
                _, _, H_out, W_out = lat_connect.shape
                upsampled = nn.Upsample((H_out, W_out), mode = 'nearest')(top_downs[-1])
                top_down_stage = lat_connect+upsampled
            else:
                top_down_stage = lat_connect
            top_downs.append(top_down_stage)
            
        P6 = nn.MaxPool2d(kernel_size = 1, stride = 2)(top_downs[0])
        output = []
        for out_stage in reversed(top_downs):
            out_stage = nn.Conv2d(C, C, kernel_size = 3, stride = 1, bias = False, padding = 1)(out_stage)
            out_stage = nn.BatchNorm2d(C)(out_stage)
            output.append(out_stage)
        output.append(P6)
        return output #[P2, P3, P4, P5, P6]
            
        
  
        
if __name__ == '__main__':
    H = 1280
    W = 960
    input = torch.zeros((1,3, H, W), dtype = torch.float)
    output = FPN().forward(input)
    print("Top down: ")
    for stage in output:
        print(stage.shape)