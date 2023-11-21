from .roi_pooler import ROI_Pooler
from torch import nn
from RPN.matcher import Matcher
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from structures import Instances, Boxes, pairwise_iou, ImageList
from .head_utils import add_ground_truth_to_proposals, subsample_labels, select_foreground_proposals
from ..utils.events import get_event_storage

class ROI_Head(nn.Module):
    '''
    This is the base class which contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads
    '''
    def __init__(self, 
                 *,
                 num_classes: int, #number of foreground classes (i.e. background is not included)
                 batch_size_per_image: int, #number of proposals to sample for training
                 positive_fraction: float, #fraction of positive (foreground) proposals to sample for training.
                 proposal_matcher: Matcher,
                 proposal_append_gt: bool = True, #whether to include ground truth as proposals as well
                 ):
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.num_classes = num_classes
        self.positive_fraction = positive_fraction
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt
    
    def _sample_proposals( 
        self,
        matched_idxs: torch.tensor, # a vector of length N, each is the best-matched gt index in [0, M) for each proposal.
        matched_labels: torch.tensor,  # a vector of length N, the matcher's label for each proposal.
        gt_classes: torch.tensor,  # a vector of length M.
    ) -> Tuple[torch.tensor,torch.tensor]:
        '''
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        '''
        has_gt = gt_classes.numel() > 0 #gt_classes.numel() return the number of elements of gt_classes tensor
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )
        
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim = 0)
        return sampled_idxs, gt_classes[sampled_idxs]
    
    @torch.no_grad()
    def label_and_sample_proposals(self, 
                                   proposals: List[Instances],
                                   targets: List[Instances],
                                   ) -> List[Instances]:
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)
        
        proposals_with_gt = []
        
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
        
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith('gt_') and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        
        return proposals_with_gt
    
    def forward(
        self,
        images: ImageList,
        features: List[torch.tensor], #List of N feature maps in shape CHW
        proposals: List[Instances], #List of N proposals at type Instances. i-th Instances contain object proposals for i-th input image.
                                    # Each Instances has 2 field: proposal_boxes and objectness_logits
        targets: Optional[List[Instances]] = None, #List of N Instances, i-th element is ground truth per instance annotations for the i-th input image.
                                                    #It may have the following fields:
                                                        # - gt_boxes
                                                        # - gt_classes
                                                        # - gt_masks
    ) -> Tuple[List[Instances], Dict[str, torch.tensor]]:
        raise NotImplementedError()
    
    
class Standard_ROI_Heads(ROI_Head):
    def __init__(
        self,
        *,
        box_in_features: List[str], #List of feature names to use for the box head
        box_pooler: ROI_Pooler, #Pooler to extra region features for box head
        box_head: nn.Module, #transform features to make box predictions
        box_predictor: nn.Module, #make box predictions from the features.
        mask_in_features: Optional[List[str]] = None, #list of feature names to use for mask pooler or mask head
        mask_pooler: Optional[ROI_Pooler] = None, #pooler to extract region features from image features. 
                                                #The mask head will then take region  features to make predictions
        mask_head: Optional[nn.Module] = None, #transform features to make mask predictions
        train_on_pred_boxes: bool = False, #whether to use proposal boxes or predicted boxes from the box head to train other heads
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        
        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.train_on_pred_boxes = train_on_pred_boxes
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.tensor],
        proposals: Optional[List[Instances]],
        targets: Optional[List[Instances]] = None,  
    ) -> Tuple[List[Instances], Dict[str, torch.tensor]]:
        
        del images
        if self.training:
            assert targets, 'No targets found'
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        
        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
    
    def forward_with_given_boxes(
        self,
        features: Dict[str, torch.tensor],
        instances: List[Instances],
    ) -> List[Instances]:
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        
        instances = self._forward_mask(features, instances)
        return instances
    
    def _forward_box(
        self, 
        features: Dict[str, torch.tensor],
        proposals: List[Instances],
    ):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
        
    def _forward_mask(
        self,
        features: Dict[str, torch.tensor],
        instances: List[Instances],
    ):
        if not self.mask_on:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        
        else:
            features = {f: features[f] for f in self.mask_in_features}
        
        return self.mask_head(features, instances)
            