from torch.nn import Conv2d, ConvTranspose2d, ReLU
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
import torch
from layers.wrappers import move_device_like
from layers import cat, ShapeSpec, get_norm
from structures import Instances, Boxes
from utils.events import EventStorage
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


def mask_rcnn_loss(
    pred_mask_logits: torch.tensor, #Tensor of shape BCHW, HW is height, width of the mask, 
                                    #C is the number of fg classes, B is the total number of predicted masks in all images
    instances: List[Instances], #A list of N instances, where N is the number of image in the batch. 
                                #These Instances are in 1:1 corresponding with the pred_mask_logits. 
                                #The gt labels associated with each Instances are stored in fields.
):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_mask = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "All masks must be in 1:1 ratio"
    
    gt_classes, gt_masks = [], []
    
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype = torch.int64)
            gt_classes.append(gt_classes_per_image)
        
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device= pred_mask_logits.device)
        gt_masks.append(gt_masks_per_image)
        
    if len(gt_masks) == 0:
        return pred_mask_logits.sum()*0
    gt_masks = cat(gt_masks, dim = 0)
    
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else: 
        indices = torch.arange(total_num_mask)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
    
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        gt_masks_bool = gt_masks > 0.5
        
    gt_masks = gt_masks.to(dtype = torch.float32)
    
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item()/max(mask_incorrect.numel(), 1.0))
    
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item()/max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)
    
    # with EventStorage() as storage:
    #     storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    #     storage.put_scalar("mask_rcnn/false_positive", false_positive)
    #     storage.put_scalar("mask_rcnn/false_negative", false_negative)
    
    # if vis_period > 0 and storage.iter % vis_period == 0:
    #     pred_masks = pred_mask_logits.sigmoid()
    #     vis_masks = torch.cat([pred_masks, gt_masks], axis = 2)
    #     name = "Left: mask prediction;      Right: gt masks"
    #     for idx, vis_mask in enumerate(vis_masks):
    #         vis_mask = torch.stack([vis_mask]*3, axis = 0)
    #         storage.put_image(name + f" ({idx})", vis_mask)
    
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction='mean')
    return mask_loss

def mask_rcnn_inference(
    pred_masks_logits: torch.tensor,
    pred_instances: List[Instances],
    ):
    cls_agnostic_mask = pred_masks_logits.size(1) == 1
    
    if cls_agnostic_mask:
        mask_probs_pred = pred_masks_logits.sigmoid()
    else: 
        num_masks = pred_masks_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        device = class_pred.device
        indices = move_device_like(torch.arange(num_masks, device = device), class_pred)
        mask_probs_pred = pred_masks_logits[indices, class_pred][:, None].sigmoid()
    
    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim = 0)
    
    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob
        
class Base_Mask_RCNN_Head(nn.Module):
    def __init__(self, *, loss_weights: float = 1.0):
        super().__init__()
        self.loss_weights = loss_weights
        
    def forward(
        self, 
        x, #Region features provided by ROIHeads
        instances: List[Instances]
    ):
        x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss(x, instances) * self.loss_weights}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        raise NotImplementedError

class Mask_RCNN_Conv_Upsample_Head(Base_Mask_RCNN_Head, nn.Sequential):
    def __init__(self, 
                 input_shape: Union[ShapeSpec, Tuple, int],
                 *,
                 num_classes: int,
                 conv_dims: List[int],
                 conv_norm = "",
                 use_dwise_conv = False,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims input have to be non-empty"
        
        if isinstance(input_shape, int):
            input_shape = ShapeSpec(channels=input_shape)
        elif isinstance(input_shape, list or tuple):
            input_shape = ShapeSpec(*input_shape)
        
        self.conv_norm_relus = []
        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            if use_dwise_conv:
                _conv = Conv2d(cur_channels,
                                conv_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=not conv_norm,
                                groups=cur_channels
                                )
            else:
                _conv = Conv2d(cur_channels,
                                conv_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=not conv_norm,
                                )
                
            weight_init.c2_msra_fill(_conv)
            
            if conv_norm != '':
                conv = nn.Sequential(
                    _conv,
                    get_norm(conv_norm, conv_dims),
                    ReLU(),
                    )
            else:
                conv = nn.Sequential(
                    _conv,
                    ReLU(),
                    )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim
        
        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", ReLU())
        cur_channels = conv_dims[-1]
        
        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
            
        weight_init.c2_msra_fill(self.deconv)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std = 0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)
    
    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x
                