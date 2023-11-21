import torch
from typing import List, Tuple, Union
from structures import Instances, Boxes
import math

def nonzero_tuple(x):
    if x.dim() == 0:
        return x.unsqueeze(0).nonzero().unbind(1)
    return x.nonzero().unbind(1)

def subsample_labels(labels: torch.Tensor, #(N, ) label vector with value 
                                                        #-1: ignore, 
                                                        # #bg_label: background, 
                                                        # #o.w: foreground
                         num_samples: int, #the total number of background and foreground labels to return.
                         positive_fraction: float, #
                         bg_label: int,
                         ):
    '''Randomly sample a subset of positive and negative samples,
    and overwrite the label vector to ignore value -1 for all eles that are not included in the sample'''
    foregrounds = nonzero_tuple((labels != -1)& (labels!=bg_label))[0]
    backgrounds = nonzero_tuple((labels == bg_label))[0]
    
    num_fg = min(foregrounds.numel(), int(positive_fraction*num_samples))
    num_bg = min(backgrounds.numel(), num_samples - num_fg)
    
    # randomly select positive and negative examples
    perm1 = torch.randperm(foregrounds.numel(), device=foregrounds.device)[:num_fg]
    perm2 = torch.randperm(backgrounds.numel(), device=backgrounds.device)[:num_bg]
    
    fg_idx = foregrounds[perm1]
    bg_idx = backgrounds[perm2]
    
    return fg_idx, bg_idx

def add_ground_truth_to_proposals(
    gt: Union[List[Instances], List[Boxes]], #List of N eles, ith ele is an Instances representing the gt for ith image
    proposals: List[Instances], #List of N eles, ith ele is an Instances representing the proposals for ith image
) -> List[Instances]:
    
    assert gt is not None
    
    if len(proposals) != len(gt):
        raise ValueError("Proposals and gt should have the same length as the number of images per batch!")
    return [add_ground_truth_to_proposals_single_image(gt_i, proposals_i) 
            for gt_i, proposals_i in zip(gt, proposals)]

def add_ground_truth_to_proposals_single_image(
    gt: Union[Instances, Boxes],
    proposals: Instances,
) -> Instances:
    if isinstance(gt, Boxes):
        gt = Instances(proposals.image_size, gt_boxes = gt)
    
    gt_boxes = gt.gt_boxes
    device = proposals.objectness_logits.device
    #Logits value of gt objects are ~=1 so assume that sigmoid(gt_logit_value) = 0.9. 
    #Then: sigmoid(gt_logit_value) = 1/(1+exp(-gt_logit_value)) = 0.9 --> gt_logit_value = ln(0.9/0.1)
    gt_logit_value = math.log((1.0 - 1e-10)/(1-(1.0 - 1e-10))) #ln(0.9/0.1)
    gt_logits = gt_logit_value*torch.ones(len(gt_boxes), device = device) #Assign this logit value to all gt objects
    
    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size, **gt.get_fields())
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits    
    
    for key in proposals.get_fields().keys():
        assert gt_proposal.has(
            key
        ), "The attribute '{}' in `proposals` does not exist in `gt`".format(key)
    
    new_proposals = Instances.cat([proposals, gt_proposal])
    return new_proposals

def select_foreground_proposals(
    proposals: List[Instances],
    bg_label: int,
) -> Tuple[List[Instances], List[Instances]]:
    
    assert isinstance(proposals, (list, tuple)), " `proposals` value must be a list or a tuple"
    assert isinstance(proposals[0], Instances), " Each proposal must be an Instances"
    assert proposals[0].has("gt_classes"), "Each proposal must contains gt_classes field"
    
    fg_proposals = []
    fg_selection_masks = []
    for proposal_per_image in proposals:
        gt_classes = proposal_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposal_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks
    
    