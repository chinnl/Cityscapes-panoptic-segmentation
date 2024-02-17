from torch import nn
import torch.nn.functional as F
import numpy as np
from layers import ShapeSpec, get_norm
from utils.postprocess import sem_seg_postprocess
from typing import Dict, List, Tuple, Optional, Union, Callable
import fvcore.nn.weight_init as weight_init

class SemSeg_FPN_Head(nn.Module):
    def __init__(self,
                 input_shape: Dict[str, ShapeSpec],
                 *,
                 num_classes: int,
                 conv_dims: int,
                 common_stride: int,
                 loss_weight: float = 1.0,
                 norm: Optional[Union[str, Callable]] = None,
                 ignore_value: int = -1,):
        super().__init__()
        input_shape = sorted(input_shape.items(), key = lambda x: x[1].stride)
        if not len(input_shape):
            raise ValueError("SemSeg_FPN_Head input_shape cannot be empty!")
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]
        
        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight

        self.scale_heads = []
        for in_feature, stride, channels in zip(
            self.in_features, feature_strides, feature_channels
        ):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv_ = nn.Conv2d(
                        channels if k == 0 else conv_dims,
                        conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not norm,
                        groups = conv_dims
                        )
                
                weight_init.c2_msra_fill(conv_)
                
                if norm_module is not None:
                    conv = nn.Sequential(
                        conv_,
                        norm_module,
                        nn.LeakyReLU(),)
                else:
                    conv = nn.Sequential(
                        conv_,
                        nn.LeakyReLU(),)
                    
                
                head_ops.append(conv)
                if stride != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = nn.Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)    
        weight_init.c2_msra_fill(self.predictor)
        
    def forward(self, features, targets = None):
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode='bilinear', align_corners=False
            )
            return x, {}
    
    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        return x
    
    def losses(self,predictions, targets):
        predictions = predictions.float()
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode= 'bilinear',
            align_corners=False
        )
        loss = F.cross_entropy(
            predictions, targets, reduction='mean', ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss*self.loss_weight}
        return losses
        