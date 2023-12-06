from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import math
from torchvision.ops import RoIPool
from structures import Boxes
from layers import ROIAlign, cat, nonzero_tuple, shapes_to_tensor, ROIAlignRotated

def assign_boxes_to_levels(
    box_lists: List[Boxes],
    min_level: int,
    max_level: int,
    canonical_box_size: int = 224,
    canonical_level:int = 4,
):
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes/canonical_box_size + 1e-8))
    #In the original scripts, they use `math.log2`. But it results in a ValueError: "Only one element tensors can be converted to Python scalars" so I changed to `torch.log2` for tensor inputs.
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level) #All the element < min_level will be set to min_level
                                                                                     #Same as all the element > max_level
    return level_assignments.to(torch.int64) - min_level

def create_zero(batch_target: Optional[torch.tensor],
                channels:int,
                height: int,
                width: int,
                like_tensor: torch.tensor,) -> torch.tensor: 
    batches = batch_target.shape[0] if batch_target is not None else 0
    sizes = (batches, channels, height, width)
    return torch.zeros(sizes, dtype=like_tensor.dtype, device=like_tensor.device)

def convert_boxes_to_pooler_format(box_lists: List[Boxes]):
    '''
    Input: box_lists: each element is an anchor boxes for each image in a batch.
    Len of box_lists is equal to batch_size.
    This function assign feature maps' size to corresponding anchor boxes (input format for ROIPool):
    Ex: [(index, boxes), ...]
    '''
    boxes = torch.cat([x.tensor for x in box_lists], dim = 0)
    sizes = shapes_to_tensor([x.__len__() for x in box_lists])
    sizes = sizes.to(device = boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(sizes), dtype = boxes.dtype, device = boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim = 1)

class ROI_Pooler(nn.Module):
    def __init__(self,
                 output_size: (int, Tuple[int] or List[int]),
                 scales: List[float], # Map scale (defined as 1 / stride)
                 sampling_ratio: int,
                 pooler_type: str,
                 canonical_box_size:int = 224,
                 canonical_level: int = 4):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        
        if pooler_type == 'ROIAlign':
            self.level_poolers = nn.ModuleList(
                ROIAlign(output_size=output_size,
                         spatial_scale=scale,
                         sampling_ratio=sampling_ratio,
                         aligned=False
                         ) for scale in scales
            )
        elif pooler_type == 'ROIAlignV2':
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size=output_size,
                    spatial_scale=scale,
                    sampling_ratio=sampling_ratio,
                    aligned=True
                ) for scale in scales
            )
        elif pooler_type == 'ROIPool':
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size=output_size, spatial_scale=scale) for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level)), "Feature map stride must be a power of 2"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        
        assert 0 <= self.min_level and self.min_level <= self.max_level, "Min_level must be >= 0 and <= max_level"
        
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size
    
    def forward(self,
                x: List[torch.tensor], #List of feature maps in NCHW shape
                box_lists: List[Boxes]): #List of N boxes (N = batch_size)
        '''
        The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.
        '''
        num_level_assignments = len(self.level_poolers)
        
        assert len(x) == num_level_assignments, "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
                num_level_assignments, len(x)
            )
        assert len(box_lists) == x[0].size(0), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
                x[0].size(0), len(box_lists)
            )
        if len(box_lists) == 0:
            return create_zero(None, x[0].shape[1], *self.output_size, x[0])
        
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        
        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        output = create_zero(pooler_fmt_boxes, num_channels, output_size, output_size, x[0]) #Output is a tensor of shape NxCxHxW 
        
        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))
        
        return output
            

