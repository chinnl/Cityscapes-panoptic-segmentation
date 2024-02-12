import torch.nn as nn
from math import ceil, log2
from torchvision.ops import StochasticDepth

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # Coefficients:   width,depth,res,dropout
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish
        # self.swish = MemoryEfficientSwish()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        stochastic_depth_prob=0.8,  # for stochastic depth
    ):
        super(MBConv, self).__init__()
        # self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        if in_channels != hidden_dim:
            layers.append(CNNBlock(
                in_channels,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ))

        layers.extend(
            [CNNBlock(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            ),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            ] )
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')

    # def stochastic_depth(self, x):
    #     if not self.training:
    #         return x

    #     binary_tensor = (
    #         torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
    #     )
    #     return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.block(inputs)

        if self.use_residual:
            return self.stochastic_depth(x) + inputs
        else:
            return x


class EfficientNet(nn.Module):
    def __init__(self, version):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, _ = self.calculate_factors(version) #The  dropout coef here is used for classifier layer
        last_channels = ceil(1280 * width_factor)
        self.features = self.create_features(width_factor, depth_factor, last_channels)

    def calculate_factors(self, version):
        width_factor, depth_factor, _, dropout = phi_values[version]
        return width_factor, depth_factor, dropout

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels
        total_stage_blocks = sum([repeats for _, _, repeats, _, _ in base_model])
        stage_block_id = 0
        stochastic_depth_prob = 0.2
        
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            stages = []
            for layer in range(layers_repeats):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stages.append(
                    MBConv(
                        in_channels,
                        out_channels,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                        expand_ratio=expand_ratio,
                        stochastic_depth_prob=sd_prob
                    )
                )
                stage_block_id += 1
                in_channels = out_channels
                
            features.append(nn.Sequential(*stages))
            
        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        output = {}
        w = x.shape[-1]
        for stage_id, block in enumerate(self.features):
            x = block(x)
            if stage_id in [2,3,5] or stage_id == len(self.features) - 1:
                output.update({f"P{int(log2(w/x.shape[-1]))}": x})
        return output
    
    @property
    def device(self):
        return next(self.parameters()).device

