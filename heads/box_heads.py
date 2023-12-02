import numpy as np
from typing import List, Union, Dict, Optional, Callable, Tuple
import torch
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, Flatten, Linear
import torch.nn as nn
from layers import ShapeSpec, get_norm, cat, cross_entropy, batched_nms
from utils.events import get_event_storage
from layers.wrappers import nonzero_tuple
from RPN.box_regression import _dense_box_regression_loss
from structures import Instances, Boxes


class Fast_RCNN_Conv_FC_Head(nn.Sequential):
    def __init__(self,
                 input_shape: ShapeSpec,
                 *,
                 conv_dims: List[int],
                 fc_dims: List[int],
                 conv_norm = ""):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.conv_norm_relus = []
        
        for k, conv_dim in enumerate(conv_dims):
            conv = nn.Sequential(
                Conv2d(self._output_size[0],
                       conv_dim,
                       kernel_size=3,
                       padding=1,
                       bias=not conv_norm),
                get_norm(conv_norm, conv_dim),
                ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        
        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", Flatten())
            fc = Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim
        
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

class Fast_RCNN_Output_Layers(nn.Module):
    def __init__(self,
                 input_shape: ShapeSpec,
                 *,
                 box2box_transform,
                 num_classes: int,
                 test_score_thresh: float = 0.0,
                 test_nms_thresh: float = 0.5,
                 test_topk_per_image: int = 100,
                 cls_agnostic_bbox_reg: bool = False,
                 smooth_l1_beta: float = 0.0,
                 box_reg_loss_type: str = "smooth_l1", #Box regression loss type. One of: "smooth_l1", "giou","diou", "ciou"
                 loss_weight: Union[float, Dict[str, float]], #A single float for all losses or: 
                                                        #* "loss_cls": applied to classification loss
                                                        # * "loss_box_reg": applied to box regression loss
                 use_fed_loss: bool = False, #whether to use federated loss which samples additional negative classes to calculate the loss
                 use_sigmoid_ce: bool = False,
                 get_fed_loss_cls_weights: Optional[Callable] = None,
                 fed_loss_num_classes: int = 50,
                 ):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = ShapeSpec(channels=input_shape)
        
        self.num_classes = num_classes
        input_size = input_shape.channels*(input_shape.width or 1)*(input_shape.height or 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes*box_dim)
        
        #Init weight with normal distributed and bias = 0
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)
        
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        
        if isinstance(loss_weight, float):
            loss_weight = {
                "loss_cls": loss_weight,
                "loss_box_reg": loss_weight
            }
        self.loss_weights = loss_weight
        self.use_fed_loss = use_fed_loss
        self.use_sigmoid_ce = use_sigmoid_ce
        self.fed_loss_num_classes = fed_loss_num_classes
        
        if self.use_fed_loss:
            assert self.use_sigmoid_ce, "Only use Sigmoid CE Loss with federated loss"
            fed_loss_cls_weights = get_fed_loss_cls_weights()
            assert (
                len(fed_loss_cls_weights) == self.num_classes
            ), "len(fed_loss_cls_weights) ({}) must equal to num_classes ({})".format(len(fed_loss_cls_weights), self.num_classes)
    
    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim = 1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
    
    def losses(self, predictions, proposals):
        scores, proposals_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim = 0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)
        
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes for p in proposals], dim = 0)
            assert not proposal_boxes.requires_grad, "Proposal boxes should not require gradient"
            
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals], dim = 0
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0,4), device = proposals_deltas.device)
        
        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = cross_entropy(scores, gt_classes, reduction = "mean")    
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposals_deltas, gt_classes
            ),
        }    
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    def sigmoid_cross_entropy_loss(self,
                                   pred_logits,
                                   gt_classes):
        if pred_logits.numel() == 0:
            return pred_logits.new_zeros([1])[0]
        
        N = pred_logits.shape[0]
        K = pred_logits.shape[1] - 1
        
        target = pred_logits.new_zeros(N, K+1)
        target[range(len(gt_classes)), gt_classes] = 1
        target = target[:, :K]
        
        cls_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_logits[:, :-1], target, reduction="none"
        )
        
        if self.use_fed_loss:
            fed_loss_classes = self.get_fed_loss_classes(
                gt_classes,
                num_fed_loss_classes = self.fed_loss_num_classes,
                num_classes = K,
                weight = self.fed_loss_cls_weights
            )
        
    def get_fed_loss_classes(self,
                             gt_classes,
                             num_fed_loss_classes,
                             num_classes,
                             weight
                             ):
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes+1).float() #create a new 'ones' tensor with 'num_classes+1' float elements
        prob[-1] = 0
        
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes
        
    def box_reg_loss(self,
                     proposal_boxes,
                     gt_boxes,
                     pred_deltas,
                     gt_classes,):
        box_dim = proposal_boxes.shape[1]
        fg_inds = nonzero_tuple((gt_classes>=0)&(gt_classes<self.num_classes))
        if pred_deltas.shape[1] == box_dim:
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
        
        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta
        )
        return loss_box_reg/max(gt_classes.numel(), 1.0)
    
    def predict_boxes_for_gt_classes(self, 
                                     predictions, #Output of self.forward()
                                     proposals, #Contain proposal_boxes and gt_classes
                                     ) -> List[torch.tensor]: #Predicted boxes for gt classes in case of class-specific box head
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim = 0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )
        
        K = predict_boxes.shape[1]//B
        if K>1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim = 0)
            gt_classes = gt_classes.clamp_(0, K-1)
            predict_boxes = predict_boxes.view(N,K,B)[
                torch.arange(N, dtype = torch.long, device=predict_boxes.device),
                gt_classes
            ]
        
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)
        
    def inference(self,
                  predictions: Tuple[torch.tensor, torch.tensor],
                  proposals: List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image
        )
        
    def predict_boxes(self,
                      predictions: Tuple[torch.tensor, torch.tensor],
                      proposals: List[Instances],):
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes for p in proposals], dim = 0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes
        )
        return predict_boxes.split(num_prop_per_image)
    
    def predict_probs(self,
                      predictions: Tuple[torch.tensor, torch.tensor],
                      proposals: List[Instances],):
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim = 1)
        return probs.split(num_inst_per_image)
        
def fast_rcnn_inference(
    boxes: List[torch.tensor],
    scores: List[torch.tensor],
    image_shapes: List[Tuple[int,int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    ):
    #Call `fast_rcnn_inference_single_image` for all images.
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, 
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image
        ) for boxes_per_image, scores_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]        
    #Return topk confident prediction and corresponding boxes/scores index in [0, Ri] from the input, for image i
    #Ri is the number of predicted objects
    return [x[0] for x in result_per_image],  [x[1] for x in result_per_image] 

def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int,int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    ):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    
    scores = scores[:,:,-1]
    num_bboxes_reg_classes = boxes.shape[1]//4 #Boxes in shape (Ri, K*4) if class-specified prediction, else (Ri, 4) for class-agnostic one
    
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bboxes_reg_classes, 4) # R*C*4
    
    #Filter out predictions with scores lower than score thresh
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()
    if num_bboxes_reg_classes == 1:
        boxes = boxes[filter_inds[:,0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >=0:
        keep = keep[:topk_per_image]
    
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

def _log_classification_stats(pred_logits, 
                              gt_classes,
                              prefix = 'fast_rcnn'):
    
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_id = pred_logits.shape[1] - 1
    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_id)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]
    
    num_false_negative = (fg_pred_classes == bg_class_id).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
    
    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)

    