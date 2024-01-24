import torch
import torch.nn as nn
from .rpn_utils import *
from .box_regression import Box2BoxTransform, _dense_box_regression_loss
from .matcher import Matcher
from typing import List, Tuple, Dict

from Backbone import FPN, R50
from layers import cat
from structures import ImageList, Instances, pairwise_iou

C = 256

class RPN_Head(nn.Module):
    '''
    Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        where:  L: number of feature maps,
                N: batch size,
                A; number of cell anchors
    '''
    def __init__(self, box_dims, num_cell_anchors) -> None:
        super(RPN_Head, self).__init__()
        
        self.feature_emb_conv = nn.Conv2d(C, C, kernel_size = 3, bias = False, padding = 1)
        self.norm1 = nn.BatchNorm2d(C)
        
        self.objness_logit_conv = nn.Conv2d(C, num_cell_anchors, kernel_size = 1, bias = False)

        self.anchor_delta_conv = nn.Conv2d(C, box_dims*num_cell_anchors, kernel_size = 1, bias = False)
        
    def forward(self, input_features):
        #input_features: list of feature maps
        pred_objectness_logits = []
        pred_anchor_deltas = []
        emb_features = []
        for _, feature_map in input_features.items():
            emb_feature = self.feature_emb_conv(feature_map)
            emb_feature = self.norm1(emb_feature)
            emb_features.append(emb_feature)
            
        pred_objectness_logits = self.get_objectness_logit(emb_features)
        
        pred_anchor_deltas = self.get_anchor_delta(emb_features)
        return pred_objectness_logits, pred_anchor_deltas
    
    
    def get_objectness_logit(self, features):
        objectness_logits = []
        for feature in features:
            obj_logit = self.objness_logit_conv(feature)
            obj_logit = nn.ReLU()(obj_logit)
            objectness_logits.append(obj_logit)
        return objectness_logits
    
    def get_anchor_delta(self, features):
        pred_anchor_deltas = []
        for feature in features:
            feature = self.anchor_delta_conv(feature)
            feature = nn.ReLU()(feature)
            pred_anchor_deltas.append(feature)
        return pred_anchor_deltas

class RPN(nn.Module):
    def __init__(self, 
                 *,
                 head: nn.Module, 
                 anchor_generator: nn.Module, 
                 anchor_matcher: Matcher, #label the anchors by matching them with ground truth.
                 batch_size_per_image: int, #number of anchors per image to sample for training
                 positive_fraction: float, #fraction of foreground anchors to sample for training
                 pre_nms_topk: Tuple[float, float], #the number of top k proposals to select
                                                    #BEFORE NMS, in training and testing.
                 post_nms_topk: Tuple[float, float], #the number of top k proposals to select
                                                    #AFTER NMS, in training and testing.
                 nms_thresh: float,
                 loss_weight: Dict[str, float], #weight for loss_rpn_cls, loss_rpn_loc
                 min_box_size: float = 0.0,
                 box2box_transform: Box2BoxTransform, #defines the transform from anchors boxes to instance boxes
                 box_reg_loss_type: str = 'smooth_l1', #"smooth_l1" or "giou"
                 smooth_l1_beta: float=0.0, #beta parameter for the smooth L1 regression loss. 
                                            #Only used when `box_reg_loss_type` is "smooth_l1"
                ):
        super().__init__()
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.loss_weight = loss_weight
        self.min_box_size = min_box_size
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.box2box_transform = box2box_transform
        
    def forward(self, 
                images: ImageList, 
                features: Dict[str, torch.tensor], 
                gt_instances: Instances):
        anchors = self.anchor_generator.forward(features) #List of L Boxes where
                                                        #each Boxes element is a tensor in shape 
                                                        #(A*Hi*Wi,4)
        anchors = [anchor_boxes.to(self.device) for anchor_boxes in anchors]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head.forward(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes)
        else:
            losses = {}
        
        proposals = self.predict_proposals(
            anchors, #(A*Hi*Wi,4)
            pred_objectness_logits, #(N, Hi*Wi*A)
            pred_anchor_deltas, #(N, Hi*Wi*A, B)
            images.image_sizes 
        )
        return proposals, losses
    
    def label_and_sample_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        gt_boxes = [gt.gt_boxes for gt in gt_instances]
        
        # image_sizes = [gt.image_size for gt in gt_instances]
        del gt_instances
        
        gt_labels = []
        matched_gt_boxes = []
        
        for gt_box in gt_boxes:
            iou_matrix = pairwise_iou(gt_box, anchors)
            matched_idxs, gt_label = self.anchor_matcher(iou_matrix)
            gt_label = gt_label.to(device=gt_box.device) #Uncomment when utilizing GPU
            del iou_matrix
            gt_label = self.subsample_labels(gt_label,
                                             self.batch_size_per_image,
                                             self.positive_fraction,
                                             0)
            if len(gt_box) == 0:
                matched_gt_box = torch.zeros_like(anchors.tensor)
            else:
                matched_gt_box = gt_box[matched_idxs].tensor
                
            gt_labels.append(gt_label)
            matched_gt_boxes.append(matched_gt_box)
        return gt_labels, matched_gt_boxes
    
    def subsample_labels(self, labels: torch.Tensor, #(N, ) label vector with value 
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
        
        labels.fill_(-1)
        labels.scatter_(0, fg_idx, 1)
        labels.scatter_(0, bg_idx, 0)
        
        return labels
    
    def losses(self, anchors: List[Boxes], 
               pred_objectness_logits: List[torch.Tensor], 
               gt_labels: List[torch.Tensor], 
               pred_anchor_deltas: List[torch.Tensor], 
               gt_boxes: List[torch.Tensor],
               ):
        '''
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        '''
        
        gt_labels = torch.stack(gt_labels) #From list of N label vector size A*Hi*Wi to tensor shape N*sum(A*Hi*Wi)
        
        fg_mask = gt_labels == 1
        
        num_images = len(gt_labels)
        # num_fg_anchors = fg_mask.sum().item()
        # num_bg_anchors = (gt_labels == 0).sum().item()
        
        localization_loss = _dense_box_regression_loss( #Only calculate loss on foreground boxes
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            fg_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )
        valid_mask = gt_labels >= 0 #Return True in loc of foreground and background boxes
        objectness_loss = nn.functional.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction = 'sum',
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
    
    @torch.no_grad() 
    def predict_proposals(self, 
                          anchors: List[Boxes], 
                          pred_objectness_logits: List[torch.Tensor], 
                          pred_anchor_deltas: List[torch.Tensor], 
                          image_sizes: List[Tuple[int, int]], 
                          ):
        N = pred_anchor_deltas[0].shape[0] #Batch size
        proposals = []
        '''
        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        '''
        for anchor, pred_anchor_delta in zip(anchors, pred_anchor_deltas):
            B = anchor.tensor.size(1) #B = 4 
            pred_anchor_delta = pred_anchor_delta.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchor = anchor.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposal = self.box2box_transform.apply_deltas(pred_anchor_delta, anchor)
            proposals.append(proposal.view(N, -1, B))
        return find_top_rpn_proposal(
            proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_size,
            self.training,
        )
        
    @property
    def device(self):
        return next(self.parameters()).device
            
if __name__ == "__main__":
    backbone = R50()
    fpn = FPN(backbone)
    rpn_head = RPN_Head(box_dims=4, num_cell_anchors=3)
    H = 800
    W = 600
    input = torch.zeros((2,3, H, W), dtype = torch.float)
    feature_maps = fpn.forward(input)
    pred_objectness_logits, pred_anchor_deltas = rpn_head.forward(feature_maps)
    print(len(pred_anchor_deltas), pred_anchor_deltas[0].shape)
    print(len(pred_objectness_logits), pred_objectness_logits[0].shape)
    
    
        