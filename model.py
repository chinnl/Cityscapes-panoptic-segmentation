from Backbone import R50, FPN
from RPN import RPN, RPN_Head, Anchor_Generator, Matcher, Box2BoxTransform
from heads import Fast_RCNN_Conv_FC_Head, Fast_RCNN_Output_Layers, Mask_RCNN_Conv_Upsample_Head, Standard_ROI_Heads, ROI_Pooler
from structures import Instances, Boxes, ImageList, BitMasks
from torch import nn 
import torch
from typing import Tuple, Dict
import numpy as np
from layers import ShapeSpec

class Mask_RCNN_FPN_R50(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, int],
                 anchor_areas: Tuple[int],
                 anchor_ratios: Tuple[int],
                 anchor_strides: Tuple[int],
                 batch_size_per_img: int,
                 positive_fraction: float,
                 pre_nms_topks: Tuple[float, float],
                 post_nms_topks: Tuple[float, float],
                 nms_thresh: float,
                 rpn_loss_weight: Dict[str, float],
                 min_box_size: float,
                 box_reg_loss_type: str,
                 is_training:bool,):
        super().__init__()
        self.is_training = is_training
        self.backbone = R50()
        
        #Because FPN is initialized based on the channels of each feature level so a dummy input is fed into the backbone for dummy feature maps.
        dummy_data = torch.zeros((1,3,input_shape[0], input_shape[1]), dtype = torch.float32)
        self.FPN = FPN(self.backbone.forward(dummy_data))
        
        rpn_head = RPN_Head(4, 3)
        anchor_gen = Anchor_Generator(anchor_areas, anchor_ratios, anchor_strides)
        anchor_matcher = Matcher([0.3, 0.5], [0, -1, 1])
        box2box_transform = Box2BoxTransform((1.0, 1.0, 1.0, 1.0)) #equal scaling factors for all level
        self.RPN = RPN(head=rpn_head,
                       anchor_generator=anchor_gen,
                       anchor_matcher=anchor_matcher,
                       batch_size_per_image=batch_size_per_img,
                       positive_fraction=positive_fraction,
                       pre_nms_topk=pre_nms_topks,
                       post_nms_topk=post_nms_topks,
                       nms_thresh=nms_thresh,
                       loss_weight=rpn_loss_weight,
                       min_box_size=min_box_size,
                       box2box_transform=box2box_transform,
                       box_reg_loss_type=box_reg_loss_type,
                       )
        
        box_pooler = ROI_Pooler(output_size=(7,7),
                                     scales=[1.0/4, 1.0/8, 1.0/16, 1.0/32],
                                     sampling_ratio=0,
                                     pooler_type='ROIAlignV2'
                                     )
        box_head = Fast_RCNN_Conv_FC_Head(input_shape=ShapeSpec(256,7,7),
                                          conv_dims=[],
                                          fc_dims=[1024,1024])
        box_predictor = Fast_RCNN_Output_Layers(input_shape=ShapeSpec(channels=1024),
                                                test_score_thresh=0.05,
                                                box2box_transform=Box2BoxTransform(weights=(10,10,5,5)),
                                                num_classes=num_classes,
                                                )
        
        mask_pooler = ROI_Pooler(output_size=14,
                                 scales = [1.0/4, 1.0/8, 1.0/16, 1.0/32],
                                 sampling_ratio=0,
                                 pooler_type='ROIAlignV2')
        mask_head = Mask_RCNN_Conv_Upsample_Head(input_shape=ShapeSpec(channels=256, width=14, height=14),
                                                 num_classes=num_classes,
                                                 conv_dims=[256, 256, 256, 256, 256],
                                                 conv_norm="")
        
        self.roi_heads = Standard_ROI_Heads(num_classes = num_classes,
                                            batch_size_per_image = batch_size_per_img,
                                            positive_fraction = 0.25,
                                            proposal_matcher = Matcher(thresholds=[0.5],
                                                                       labels=[0,1],
                                                                       allow_low_quality_matches=False),
                                            box_in_features=["P2", "P3", "P4", "P5"],
                                            box_pooler=box_pooler,
                                            box_head=box_head,
                                            box_predictor=box_predictor,
                                            mask_in_features=["P2", "P3", "P4", "P5"],
                                            mask_pooler=mask_pooler,
                                            mask_head=mask_head,
                                            )
        
        
    def forward(self, 
                batch, 
                is_training: bool = False):
        '''
        a batch is expected to contains:
            - batch_images: tensor at shape (B, C, H, W) with each element is an image in batch.
            - batch_gt: List[Instances] contains ground truth for each image in batch
        '''
        batch_images, batch_gt = batch
        if is_training:
            assert batch_gt, "If the model is being trained, batch_gt mustn't be None"
        image_list = ImageList(batch_images, [(image.shape[-2], image.shape[-1]) for image in batch_images])
        
        features = self.backbone.forward(batch_images)
        features = self.FPN.forward(features)
        
        self.RPN.training = is_training
        proposals, rpn_losses = self.RPN.forward(image_list, features, batch_gt)
        
        self.roi_heads.training = is_training
        predictions, roi_head_losses = self.roi_heads.forward(image_list, features, proposals, batch_gt)
        
        if is_training:
            total_losses = {}
            total_losses.update(rpn_losses)
            total_losses.update(roi_head_losses)
            return predictions, total_losses
        else:
            return predictions, {}


    def num_params(self):
        num_params = 0.0
        for p in list(self.parameters()):
            num_params+=p.view(-1).shape[0]
        return num_params

if __name__ == "__main__":
    model = Mask_RCNN_FPN_R50(
        num_classes=1,
        input_shape=(600,800),
        anchor_areas=(32**2,64**2,128**2,256**2,512**2),
        anchor_ratios=(0.5,1.0,2.0),
        anchor_strides = (4,8,16,32,64),
        batch_size_per_img=128,
        positive_fraction=0.5,
        pre_nms_topks=[8000, 1000],
        post_nms_topks=[1000,500],
        nms_thresh=0.2,
        rpn_loss_weight={
                    "loss_rpn_cls": 0.3,
                    "loss_rpn_loc": 0.5
                    },
        min_box_size=0.0,
        box_reg_loss_type='giou',
        is_training=True
    )
    print('Total number of params: ', model.num_params())
    import cv2
    org_img = cv2.cvtColor(cv2.imread(r"E:\20231\DATN\Cityscapes-panoptic-segmentation\bill_image.jpg"), cv2.COLOR_BGR2RGB)
    dummy_input = torch.tensor(org_img.transpose(2,0,1)).unsqueeze(0)
    dummy_input = dummy_input.to(torch.float)
    # dummy_input = torch.FloatTensor(size=(1,3,600,800)).uniform_(0,255)
    
    dummy_gt = Instances(image_size=(dummy_input.shape[-2],dummy_input.shape[-1]))
    dummy_gt.gt_boxes = Boxes(torch.tensor([[78., 107., 616., 858.]]))
    dummy_gt.gt_classes = torch.tensor([0])
    # dummy_gt.gt_masks = BitMasks(torch.randint(0, 255, (1,600,800)))
    predictions, losses = model.forward((dummy_input, [dummy_gt]), False)
    
    for box in predictions[0].pred_boxes:
        cv2.rectangle(org_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 1)
    cv2.imwrite("result.jpg", cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
    print(len(predictions[0].pred_boxes))