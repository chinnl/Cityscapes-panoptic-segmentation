import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import PSRoIPool, PSRoIAlign
from typing import List, Tuple, Optional, Dict
from structures import Instances, ImageList
from layers import cat, nonzero_tuple
from .roi_pooler import convert_boxes_to_pooler_format, assign_boxes_to_levels, create_zero
from RPN.box_regression import _dense_box_regression_loss, Box2BoxTransform
from .ROI_Heads import ROI_Head
import math
from .head_utils import select_foreground_proposals
from heads.box_heads import fast_rcnn_inference
from heads.mask_heads import mask_rcnn_inference, mask_rcnn_loss
from fvcore.nn import weight_init

class RFCN(ROI_Head):
    def __init__(self,
                 *,
                 box_in_features: List[str],
                 output_size: List[int],
                 scales: List[int],
                 canonical_box_size: int = 224,
                 canonical_level: int = 4,
                 ignore_value: int = 255,
                 loss_weights: Dict[str, float],
                 box2box_transform: Box2BoxTransform,
                 box_reg_loss_type:str = 'smooth_l1',
                 test_score_thresh: float = 0.0,
                 test_nms_thresh: float = 0.5,
                 test_topk_per_image: int = 100,
                 smooth_l1_beta: float = 1.0,
                 mask_in_features: List[str] = None,
                 mask_pooler: Optional[PSRoIPool] = None, 
                 mask_head: Optional[nn.Module] = None, 
                 train_on_pred_boxes: bool = False, 
                 **kwargs):
        super().__init__(**kwargs)
        self.box_in_features = box_in_features
        self.output_size = output_size
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level
        self.ignore_value = ignore_value
        self.box2box_transform = box2box_transform
        self.box_reg_loss_type = box_reg_loss_type
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.smooth_l1_beta = smooth_l1_beta
        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.train_on_pred_boxes = train_on_pred_boxes
        self.loss_weights = loss_weights
        
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level)), "Feature map stride must be a power of 2"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        
        if isinstance(self.output_size, list):
            assert self.output_size[0] != self.output_size[1], 'ROIs must be square'
            self.output_size = output_size[0]
            
        self.generate_score_maps = nn.Conv2d(256, output_size*output_size*(self.num_classes+1), kernel_size=(1,1), bias = False)
        self.generate_box_maps = nn.Conv2d(256, output_size*output_size*4, kernel_size=(1,1), bias = False)
        weight_init.c2_msra_fill(self.generate_score_maps)
        weight_init.c2_msra_fill(self.generate_box_maps)
        
        self.predictor = nn.AvgPool2d((7,7),stride=(7,7))
        
        self.level_poolers = nn.ModuleList(
            PSRoIPool(output_size = output_size, spatial_scale=scale) for scale in scales
            )
        
    def forward(self, 
                images: ImageList,
                features: Dict[str, torch.tensor],
                proposals: List,
                targets: List):
        del images
        if self.training:
            assert targets, 'No targets found'
            proposals = self.label_and_sample_proposals(proposals, targets)
            del targets
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            return proposals, losses
        
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self._forward_mask(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        box_list = [x.proposal_boxes for x in proposals]
        
        if len(box_list) == 0:
            return create_zero(None, features[0].shape[1], self.output_size, self.output_size, features[0])
        
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_list)
        level_assignments = assign_boxes_to_levels(
            box_list, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )
        level_score_maps = [self.generate_score_maps(x) for x in features]
        level_box_maps = [self.generate_box_maps(x) for x in features]
        
        score_output = create_zero(pooler_fmt_boxes, self.num_classes + 1, self.output_size, self.output_size, features[0]) #Output is a tensor of shape NxCxHxW 
        box_output = create_zero(pooler_fmt_boxes, 4, self.output_size, self.output_size, features[0]) #Output is a tensor of shape NxCxHxW 
        
        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            score_output.index_put_((inds,), pooler(level_score_maps[level], pooler_fmt_boxes_level))
            box_output.index_put_((inds,), pooler(level_box_maps[level], pooler_fmt_boxes_level))
        
        proposals_deltas = self.predictor(box_output).squeeze()
        scores = self.predictor(score_output).squeeze()
        if self.training:
            return self.losses((proposals_deltas, scores), proposals)
        else:
            pred_instances, _ = self.predict_boxes((proposals_deltas, scores), proposals)
            return pred_instances
    
    def losses(self, predictions, proposals):
        proposals_deltas, scores = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim = 0) if len(proposals) else torch.empty(0)
        )
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim = 0)
            assert not proposal_boxes.requires_grad, "Proposal boxes should not require gradient"
            
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals], dim = 0
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0,4), device = proposals_deltas.device)
        
        loss_cls = F.cross_entropy(scores, gt_classes, ignore_index=self.ignore_value, reduction='mean')
        loss_box_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposals_deltas, gt_classes)
        losses =  {
                "loss_cls": loss_cls,
                "loss_box_reg": loss_box_reg
               }
        return {k: v * self.loss_weights.get(k, 1.0) for k, v in losses.items()}
    
    def predict_boxes(self,
                  predictions: Tuple[torch.tensor, torch.tensor],
                  proposals: List[Instances]):
        proposal_deltas, scores = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        num_prop_per_image = [len(p) for p in proposals]
        
        predicted_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes
        ).split(num_prop_per_image)
        
        predicted_probs = F.softmax(scores, dim=-1).split(num_prop_per_image, dim=0)
        image_shapes = [x.image_size for x in proposals]
        
        return fast_rcnn_inference(
            predicted_boxes,
            predicted_probs,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image
        )
        
    def box_reg_loss(self,
                     proposal_boxes,
                     gt_boxes,
                     pred_deltas,
                     gt_classes,):
        box_dim = proposal_boxes.shape[1]
        fg_inds = nonzero_tuple((gt_classes>=0)&(gt_classes<self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
        
        loss_box_reg = _dense_box_regression_loss(
            anchors=[proposal_boxes[fg_inds]],
            box2box_transform=self.box2box_transform,
            pred_anchor_deltas=[fg_pred_deltas.unsqueeze(0)],
            gt_boxes=[gt_boxes[fg_inds]],
            fg_mask=...,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta
        )
        return loss_box_reg/max(gt_classes.numel(), 1.0)
        

    def _forward_mask(
        self,
        features: Dict[str, torch.tensor],
        instances: List[Instances],
    ):
        if not self.mask_on:
            return {} if self.training else instances
            
        if self.training:    
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        
        else:
            features = {f: features[f] for f in self.mask_in_features}
        
        return self.mask_head(features, instances)
    
    def forward_with_given_boxes(
        self,
        features: Dict[str, torch.tensor],
        instances: List[Instances],
    ) -> List[Instances]:
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        
        instances = self._forward_mask(features, instances)
        return instances

class RFCN2(ROI_Head):
    def __init__(self,
                 *,
                 box_in_features: List[str],
                 output_size: List[int],
                 scales: List[int],
                 canonical_box_size: int = 224,
                 canonical_level: int = 4,
                 ignore_value: int = 255,
                 loss_weights: Dict[str, float],
                 box2box_transform: Box2BoxTransform,
                 box_reg_loss_type:str = 'smooth_l1',
                 test_score_thresh: float = 0.0,
                 test_nms_thresh: float = 0.5,
                 test_topk_per_image: int = 100,
                 smooth_l1_beta: float = 1.0,
                 train_on_pred_boxes: bool = False, 
                 loss_mask_weights: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.box_in_features = box_in_features
        self.output_size = output_size
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level
        self.ignore_value = ignore_value
        self.box2box_transform = box2box_transform
        self.box_reg_loss_type = box_reg_loss_type
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.smooth_l1_beta = smooth_l1_beta
        self.train_on_pred_boxes = train_on_pred_boxes
        self.loss_weights = loss_weights
        self.loss_mask_weights = loss_mask_weights
        
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level)), "Feature map stride must be a power of 2"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        
        if isinstance(self.output_size, list):
            assert self.output_size[0] != self.output_size[1], 'ROIs must be square'
            self.output_size = output_size[0]
            
        self.generate_score_maps = nn.Conv2d(256, output_size*output_size*(self.num_classes+1), kernel_size=(1,1), bias = True)
        self.generate_box_maps = nn.Conv2d(256, output_size*output_size*4, kernel_size=(1,1), bias = True)
        self.generate_instance_masks = nn.Conv2d(256, output_size*2*output_size*2*self.num_classes, kernel_size=(1,1), bias = True)
        weight_init.c2_msra_fill(self.generate_score_maps)
        weight_init.c2_msra_fill(self.generate_box_maps)
        weight_init.c2_msra_fill(self.generate_instance_masks)
        
        self.predictor = nn.AvgPool2d((7,7),stride=(7,7))
        self.mask_predictor = nn.AvgPool2d((14,14),stride=(14,14))
        
        self.level_poolers = nn.ModuleList(
            PSRoIPool(output_size = output_size, spatial_scale=scale) for scale in scales
            )
        
        self.mask_level_poolers = nn.ModuleList(
            PSRoIPool(output_size = output_size*2, spatial_scale=scale) for scale in scales
            )
        
        self.deconv = nn.ConvTranspose2d(
            self.num_classes, self.num_classes, kernel_size=2, stride=2, padding=0
        )
        
        
    def forward(self, 
                images: ImageList,
                features: Dict[str, torch.tensor],
                proposals: List,
                targets: List):
        del images
        if self.training:
            assert targets, 'No targets found'
            proposals = self.label_and_sample_proposals(proposals, targets)
            del targets
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            return proposals, losses
        
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self._forward_mask(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        box_list = [x.proposal_boxes for x in proposals]
        
        if len(box_list) == 0:
            return create_zero(None, features[0].shape[1], self.output_size, self.output_size, features[0])
        
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_list)
        level_assignments = assign_boxes_to_levels(
            box_list, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )
        level_score_maps = [self.generate_score_maps(x) for x in features]
        level_box_maps = [self.generate_box_maps(x) for x in features]
        
        
        score_output = create_zero(pooler_fmt_boxes, self.num_classes + 1, self.output_size, self.output_size, features[0]) 
        box_output = create_zero(pooler_fmt_boxes, 4, self.output_size, self.output_size, features[0]) 
        
        
        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            score_output.index_put_((inds,), pooler(level_score_maps[level], pooler_fmt_boxes_level))
            box_output.index_put_((inds,), pooler(level_box_maps[level], pooler_fmt_boxes_level))
            
        proposals_deltas = self.predictor(box_output).squeeze()
        scores = self.predictor(score_output).squeeze()
        
        if self.training:
            losses = self.losses((proposals_deltas, scores), proposals)
            return losses
        else:
            pred_instances, _ = self.predict_boxes((proposals_deltas, scores), proposals)
            return pred_instances
    
    def losses(self, predictions, proposals):
        proposals_deltas, scores = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim = 0) if len(proposals) else torch.empty(0)
        )
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim = 0)
            assert not proposal_boxes.requires_grad, "Proposal boxes should not require gradient"
            
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals], dim = 0
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0,4), device = proposals_deltas.device)
        
        loss_cls = F.cross_entropy(scores, gt_classes, ignore_index=self.ignore_value, reduction='mean')
        loss_box_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposals_deltas, gt_classes)
        losses =  {
                "loss_cls": loss_cls,
                "loss_box_reg": loss_box_reg
               }
        return {k: v * self.loss_weights.get(k, 1.0) for k, v in losses.items()}
    
    def predict_boxes(self,
                  predictions: Tuple[torch.tensor, torch.tensor],
                  proposals: List[Instances]):
        proposal_deltas, scores = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        num_prop_per_image = [len(p) for p in proposals]
        
        predicted_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes
        ).split(num_prop_per_image)
        
        predicted_probs = F.softmax(scores, dim=-1).split(num_prop_per_image, dim=0)
        image_shapes = [x.image_size for x in proposals]
        
        return fast_rcnn_inference(
            predicted_boxes,
            predicted_probs,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image
        )
        
    def box_reg_loss(self,
                     proposal_boxes,
                     gt_boxes,
                     pred_deltas,
                     gt_classes,):
        box_dim = proposal_boxes.shape[1]
        fg_inds = nonzero_tuple((gt_classes>=0)&(gt_classes<self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
        
        loss_box_reg = _dense_box_regression_loss(
            anchors=[proposal_boxes[fg_inds]],
            box2box_transform=self.box2box_transform,
            pred_anchor_deltas=[fg_pred_deltas.unsqueeze(0)],
            gt_boxes=[gt_boxes[fg_inds]],
            fg_mask=...,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta
        )
        return loss_box_reg/max(gt_classes.numel(), 1.0)
        

    def _forward_mask(
        self,
        features: Dict[str, torch.tensor],
        instances: List[Instances],
    ):      
        if self.training:    
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        features = [features[f] for f in self.box_in_features]
        # box_list = [x.proposal_boxes for x in instances]
        box_list = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_list)
        level_assignments = assign_boxes_to_levels(
            box_list, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )
        level_masks = [self.generate_instance_masks(x) for x in features]
        mask_output = create_zero(pooler_fmt_boxes, self.num_classes, self.output_size*2, self.output_size*2, features[0])
        for level, pooler in enumerate(self.mask_level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            mask_output.index_put_((inds,), pooler(level_masks[level], pooler_fmt_boxes_level))
        mask_logits = self.deconv(mask_output)
        
        if self.training:
            return {"loss_mask": mask_rcnn_loss(mask_logits, instances) * self.loss_mask_weights}
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances

    