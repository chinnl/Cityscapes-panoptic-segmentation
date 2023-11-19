import torch
import torch.nn as nn
import math
from typing import List, Tuple, Union
from torch import device
import numpy as np

import sys
sys.path.append(r"E:\20231\DATN\Cityscapes-panoptic-segmentation")
from structures import Boxes, Instances
from layers import cat, batched_nms


def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)

class Anchor_Generator(nn.Module):
    def __init__(self, anchor_sizes, aspect_ratios, strides, offset):
        super(Anchor_Generator, self).__init__()
        """
        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            !!!Modify: each element in sizes is anchor area to prevent rounding error
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        self.strides = strides
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.offset = offset
        
    def forward(self, input_features, scaling_factors):
        grid_sizes = [input_feature.shape[2:] for _, input_feature in input_features.items()]
        grid_anchors = self.generate_grid_anchors(grid_sizes, self.anchor_sizes, self.aspect_ratios)
        # return [Boxes(x) for x in grid_anchors]
        anchors = []
        for anchor, scaling_factor in zip(grid_anchors, scaling_factors):
            anchor = Boxes(anchor)
            anchor.scale(scaling_factor, scaling_factor)
            anchors.append(anchor)
        return anchors
        
    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        #Anchor's center is (0,0)
        if not isinstance(sizes, (list, tuple)):
            sizes = [sizes]
        if not isinstance(aspect_ratios, (list, tuple)):
            aspect_ratios = [aspect_ratios]
            
        for area in sizes:
            for ratio in aspect_ratios:
                a_w = math.sqrt(area/ratio)
                a_h = a_w*ratio
                anchors.append([-a_w/2.0, -a_h/2.0, a_w/2.0, a_h/2.0])
        return torch.tensor(anchors)
    
    def generate_grid_anchors(self, grid_sizes, anchor_sizes, aspect_ratios):
        anchors = []
        for (grid_size, anchor_size) in zip(grid_sizes, anchor_sizes):
            '''
            - grid_size = (grid_h, grid_w)
            - image_size = Tuple(int(), int()) #(H, W)
            - anchor_sizes = int (anchor area)
            - aspect_ratios = (0.5, 1.0, 2.0)
            '''
            base_anchors = self.generate_cell_anchors(
                            anchor_size, 
                            aspect_ratios) #3 base anchor with corresponding ratio 0.5, 1.0 and 2.0
            shift_x, shift_y = self.create_grid_offset(grid_size, self.strides, self.offset, base_anchors)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim = 1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors
    
    def create_grid_offset(self, grid_size, stride, offset, target_tensor):
        '''
        Create offset grid for each anchor
        '''
        grid_h, grid_w = grid_size
        shift_x = move_device_like(torch.arange(offset*stride, grid_w, 
                                                #grid_w*stride, but changed to grid_w 
                                                #because if stride>1 then the anchors' coordinates
                                                # are bigger than feature maps' sizes.
                                                step = stride, dtype = torch.float32), target_tensor)
        shift_y = move_device_like(torch.arange(offset*stride, grid_h, 
                                                #same as shift_x
                                                step = stride, dtype = torch.float32), target_tensor)
        
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

def nonzero_tuple(x):
    if x.dim() == 0:
        return x.unsqueeze(0).nonzero().unbind(1)
    return x.nonzero().unbind(1)

def find_top_rpn_proposal(
    proposals: List[torch.Tensor], #(A*Hi*Wi,4)
    pred_objectness_logits: List[torch.tensor], #(N, Hi*Wi*A)
    image_sizes: List[Tuple[int,int]], 
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
    ):
    '''
    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    '''
    num_images = len(image_sizes)
    # device = (
    #     proposals[0].device()
    # ) #Uncomment to ultilize GPU
    device = torch.device('cpu')
    #Select topk anchor for every level and image
    
    topk_scores = [] #List of N element, each of shape (pre_nms_topk,)
    topk_proposals = [] #List of N element, each of shape (pre_nms_topk, 4)
    level_ids = [] #List of N element, each of shape (pre_nms_topk, )
    batch_idx = move_device_like(torch.arange(num_images, device = device), proposals[0])
    
    for level, (proposal, logits) in enumerate(zip(proposals, pred_objectness_logits)): 
        Hi_Wi_A = logits.shape[1] #number of objects 
        if isinstance(Hi_Wi_A, torch.Tensor):
            num_proposals = torch.clamp(Hi_Wi_A, max = pre_nms_topk)
        else:
            num_proposals = min(Hi_Wi_A, pre_nms_topk)
            
        topk_score, topk_idx = logits.topk(num_proposals, dim=1)
        
        topk_proposal = proposal[batch_idx[:, None], topk_idx] #N * topk * 4
        
        topk_proposals.append(topk_proposal)
        topk_scores.append(topk_score)
        level_ids.append(
            move_device_like(
                torch.full((num_proposals,), level, dtype = torch.int64, device = device),
                proposals[0]
                )
            ) #A tensor fulled with value "level" and has shape (num_proposal,)
        
    #Concat all levels
    topk_scores = cat(topk_scores, dim =1) # (N * pre_nms_topk,)
    topk_proposals = cat(topk_proposals, dim = 1) #(N * pre_nms_topk, * 4)
    level_ids = cat(level_ids, dim = 0) #(N*pre_nms_topk,)
    
    #NMS
    result: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        level = level_ids
        
        valid_mask = torch.isfinite(boxes.tensor).all(dim=1)&torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            level = level[valid_mask]
        boxes.clip(image_size)
        
        keep = boxes.nonempty(threshold=min_box_size)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, level = boxes[keep], scores_per_img[keep], level[keep]
            
        keep = batched_nms(boxes.tensor, scores_per_img, level, nms_thresh)
        keep = keep[:post_nms_topk]
        
        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        result.append(res)
    return result



if __name__ == "__main__":
    from structures import *
    from Backbone import FPN, R50
    H = 800
    W = 600
    input = torch.zeros((2,3, H, W), dtype = torch.float)
    features = R50().forward(input)
    output = FPN(features).forward()
    anchor_generator = Anchor_Generator(anchor_sizes=[32,64,128,256,512],
                                        aspect_ratios=(0.5, 1.0, 2.0),
                                        strides=1,
                                        offset=0.5)
    anchors = anchor_generator.forward(output, (4,8,16,32,64))
    print(len(anchors))
    for anchor_list in anchors:
        print(len(anchor_list))