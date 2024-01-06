import torch
import torch.nn as nn
from typing import List, Dict
from structures import ImageList
from utils import sem_seg_postprocess, detector_postprocess
from .rcnn import GeneralizedRCNN

class PanopticFPN(GeneralizedRCNN):
    def __init__(self, 
                 *,
                 sem_seg_head: nn.Module,
                 combine_overlap_thresh: float = 0.5,
                 combine_stuff_area_thresh: float = 4096,
                 combine_instances_score_thresh: float = 0.5,
                 **kwargs):
        '''
        Args:
            sem_seg_head: a module for the semantic segmentation head.
            combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
            combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
            combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold
        '''
        super().__init__(**kwargs)
        self.sem_seg_head = sem_seg_head
        self.combine_overlap_thresh = combine_overlap_thresh
        self.combine_stuff_area_thresh = combine_stuff_area_thresh
        self.combine_instances_score_thresh = combine_instances_score_thresh
        
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        assert "sem_seg" in batched_inputs[0], "Semantic segment gt not found in inputs"
        gt_sem_seg = [x['sem_seg'].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(
            gt_sem_seg,
            self.cfg.general.padding_constraint.size_divisibility,
            self.sem_seg_head.ignore_value,
            self.cfg.general.padding_constraint
        ).tensor
        
        _, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)
        
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )
        
        losses = sem_seg_losses
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses
    
    def inference(self, 
                  batched_inputs: List[Dict[str, torch.tensor]], 
                  do_postprocess: bool = True):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        sem_seg_results, _ = self.sem_seg_head(features, None)
        proposals, _ = self.proposal_generator(images, features, None)
        detector_results, _ = self.roi_heads(images, features, proposals, None)
        
        if do_postprocess:
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes):
                
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                detector_r = detector_postprocess(detector_result, height, width)
                
                processed_results.append({"sem_seg": sem_seg_r,
                                          "instances": detector_r})
                
                panoptic_r = combine_semantic_and_instances_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim = 0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        else:
            return detector_results, sem_seg_results
    
def combine_semantic_and_instances_outputs(instance_results,
                                           semantic_results,
                                           overlap_threshold,
                                           stuff_area_thresh,
                                           instances_score_thresh):
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)
    sorted_inds = torch.argsort(-instance_results.scores)
    current_segment_id = 0
    segments_info = []
    
    instance_masks = instance_results.pred_masks.to(dtype = torch.bool, device = panoptic_seg.device)
    
    for instance_id in sorted_inds:
        score = instance_results.scores[instance_id].item()
        if score < instances_score_thresh:
            break
        
        mask = instance_masks[instance_id]
        mask_area = mask.sum().item()
        
        if mask_area == 0:
            continue
        
        intersect = (mask>0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()
        
        if intersect_area*1.0/mask_area > overlap_threshold:
            continue
        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)
        
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[instance_id].item(),
                "instance_id": instance_id.item(),
            }
        )
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    
    for semantic_label in semantic_labels:
        if semantic_label == 0:
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_thresh:
            continue
        current_segment_id +=1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                'isthing': False,
                'category_id': semantic_label,
                'area': mask_area,
            }
        )
    return panoptic_seg, segments_info