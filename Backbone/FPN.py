import torch
import torch.nn as nn


C = 256
class FPN(nn.Module):
    def __init__(self, input):
        super(FPN, self).__init__()
        self.input = input
        self.in_channels = [feature.shape[1] for _, feature in self.input.items()]
        self.lat_conv2 = nn.Sequential(
                                nn.Conv2d(self.in_channels[-1], C, kernel_size = 1, stride = 1, bias = False), 
                                nn.BatchNorm2d(C)
                                ) #P2
        self.lat_conv3 = nn.Sequential(
                                nn.Conv2d(self.in_channels[-2], C, kernel_size = 1, stride = 1, bias = False), 
                                nn.BatchNorm2d(C)
                                ) #P3
        self.lat_conv4 = nn.Sequential(
                                nn.Conv2d(self.in_channels[-3], C, kernel_size = 1, stride = 1, bias = False), 
                                nn.BatchNorm2d(C)
                                ) #P4
        self.lat_conv5 = nn.Sequential(
                                nn.Conv2d(self.in_channels[-4], C, kernel_size = 1, stride = 1, bias = False), 
                                nn.BatchNorm2d(C)
                                ) #P5
        self.output_conv = nn.Sequential(
                                nn.Conv2d(C, C, kernel_size = 3, stride = 1, bias = False, padding = 1),
                                nn.BatchNorm2d(C),
                                )
    def forward(self):
        top_downs = []
        for P_i, lat_conv in zip(self.input.values(), [self.lat_conv5, self.lat_conv4, self.lat_conv3, self.lat_conv2]):
            lat_connect = lat_conv(P_i)
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
            out_stage = self.output_conv(out_stage)
            output.append(out_stage)
        output.append(P6)
        output_dict = {}
        for feature, level in zip(output, ['P2', 'P3', 'P4', 'P5', 'P6']):
            output_dict[level] = feature
        return output_dict #[P2, P3, P4, P5, P6]
            
        
  
        
if __name__ == '__main__':
    H = 800
    W = 600
    input = torch.zeros((2,3, H, W), dtype = torch.float)
    from resnet import R50
    features = R50().forward(input)
    output = FPN(features).forward()
    for key, stage in output.items():
        print(key, stage.shape)