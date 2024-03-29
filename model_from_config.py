from Backbone import R50, FPN
from Backbone.efficient_net import EfficientNet
from RPN import RPN, RPN_Head, Anchor_Generator, Matcher, Box2BoxTransform
from heads import *
from torch import nn 
import torch
from layers import ShapeSpec
from torchvision.models import resnet50, ResNet50_Weights
from heads.rfcn import RFCN2

class Backbone(nn.Module):
    def __init__(self, bottom_up, fpn):
        super().__init__()
        self.bottom_up = bottom_up
        self.fpn = fpn
    
    def forward(self, x): 
        x = nn.Sequential(self.bottom_up, self.fpn)(x)
        return x
    @property
    def device(self):
        return next(self.parameters()).device

def load_model(cfg):
    feature_shapes = {
        k: ShapeSpec(channels=256, stride = v) 
        for k, v in zip(["P2", "P3", "P4", "P5"], [4, 8, 16, 32, 64])
    } 
    
    bottom_up = R50()
    pretrained_r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    map_r50_block(bottom_up.res2, pretrained_r50.layer1)
    map_r50_block(bottom_up.res3, pretrained_r50.layer2)
    map_r50_block(bottom_up.res4, pretrained_r50.layer3)
    map_r50_block(bottom_up.res5, pretrained_r50.layer4)
    

    dummy_data = torch.zeros((1,3,cfg.general.input_shape[0], cfg.general.input_shape[1]), dtype = torch.float32)
    fpn = FPN(bottom_up(dummy_data))
    backbone = Backbone(bottom_up, fpn)

    rpn_head = RPN_Head(4, len(cfg.anchor_generator.anchor_ratios))
    anchor_gen = Anchor_Generator(cfg.anchor_generator.anchor_sizes, 
                                    cfg.anchor_generator.anchor_ratios, 
                                    cfg.anchor_generator.anchor_strides)
       
    rpn = RPN(head=rpn_head,
            anchor_generator = anchor_gen,
            anchor_matcher = Matcher(cfg.rpn.matcher.thresholds,
                                    cfg.rpn.matcher.labels,
                                    cfg.rpn.matcher.allow_low_quality_matches),
            batch_size_per_image = cfg.rpn.batch_size_per_img,
            positive_fraction = cfg.rpn.positive_fraction,
            pre_nms_topk = cfg.rpn.pre_nms_topk,
            post_nms_topk = cfg.rpn.post_nms_topk,
            nms_thresh = cfg.rpn.nms_thresh,
            loss_weight = cfg.rpn.loss_weights,
            min_box_size = cfg.rpn.min_box_size,
            box2box_transform = Box2BoxTransform(cfg.rpn.box_transform_weights), #equal scaling factors for all level
            box_reg_loss_type = cfg.rpn.box_reg_loss_type,
            )
    # load_rpn_weights(rpn)
    
    box_pooler = ROI_Pooler(output_size = cfg.box_pooler.output_shape,
                            scales = cfg.box_pooler.scales,
                            sampling_ratio = cfg.box_pooler.sampling_ratio,
                            pooler_type = cfg.box_pooler.pooler_type,
                            )
    
    box_head = Fast_RCNN_Conv_FC_Head(input_shape = cfg.box_head.input_shape,
                                    conv_dims = cfg.box_head.conv_dims,
                                    fc_dims = cfg.box_head.fc_dims,
                                    conv_norm = cfg.box_head.conv_norm)
    
    box_predictor = Fast_RCNN_Output_Layers(input_shape = cfg.box_predictor.input_shape,
                                            test_score_thresh = cfg.box_predictor.test_score_thresh,
                                            box2box_transform = Box2BoxTransform(cfg.box_predictor.box_transform_weights),
                                            num_classes = cfg.general.num_classes,
                                            box_reg_loss_type = cfg.box_predictor.box_reg_loss_type,
                                            smooth_l1_beta = cfg.box_predictor.smooth_l1_beta,
                                            loss_weight = cfg.box_predictor.loss_weight,
                                            ignore_value = cfg.general.num_classes-1)
    
    mask_pooler = ROI_Pooler(output_size=cfg.mask_pooler.output_size,
                            scales = cfg.mask_pooler.scales,
                            sampling_ratio=cfg.mask_pooler.sampling_ratio,
                            pooler_type=cfg.mask_pooler.pooler_type)
    
    mask_head = Mask_RCNN_Conv_Upsample_Head(input_shape = cfg.mask_head.input_shape,
                                            num_classes = cfg.general.num_classes,
                                            conv_dims = cfg.mask_head.conv_dims,
                                            conv_norm = cfg.mask_head.conv_norm,
                                            loss_weights = cfg.mask_head.loss_weight)
    
    roi_heads = Standard_ROI_Heads( num_classes = cfg.general.num_classes,
                                    batch_size_per_image = cfg.roi_heads.batch_size_per_img,
                                    positive_fraction = cfg.roi_heads.positive_fraction,
                                    proposal_matcher = Matcher(thresholds=cfg.roi_heads.matcher.thresholds,
                                                                labels=cfg.roi_heads.matcher.labels,
                                                                allow_low_quality_matches=cfg.roi_heads.matcher.allow_low_quality_matches),
                                    box_in_features=cfg.roi_heads.box_in_features,
                                    box_pooler=box_pooler,
                                    box_head=box_head,
                                    box_predictor=box_predictor,
                                    mask_in_features=cfg.roi_heads.mask_in_features,
                                    mask_pooler=mask_pooler,
                                    mask_head=mask_head,
                                    train_on_pred_boxes=cfg.roi_heads.train_on_pred_boxes
                                    )
    
    sem_seg_head = SemSeg_FPN_Head( input_shape = feature_shapes,
                                    num_classes = cfg.general.num_classes,
                                    conv_dims = 128,
                                    common_stride = 4,
                                    loss_weight=cfg.semantic_head.loss_weight,
                                    norm="GN", 
                                    ignore_value = cfg.general.num_classes-1)
    
    pixel_mean = [72.5239, 83.0444, 73.2913] 
    pixel_std = [0.1860, 0.1888, 0.1856] 
    input_format="BGR"
    
    model = PanopticFPN(cfg = cfg,
                        backbone = backbone,
                        proposal_generator = rpn,
                        roi_heads = roi_heads,
                        sem_seg_head = sem_seg_head,
                        combine_overlap_thresh = cfg.panoptic_head.combine_overlap_thresh,
                        combine_stuff_area_thresh = cfg.panoptic_head.combine_stuff_area_thresh,
                        combine_instances_score_thresh = cfg.panoptic_head.combine_instances_score_thresh,
                        pixel_mean = pixel_mean,
                        pixel_std = pixel_std,
                        input_format = input_format)
    print(f"Model is initialized with {int(num_params(model))} params. . . . .")
    return model




def load_efficient_model(cfg):
    bottom_up = EfficientNet('efficientnet-b3')
    # load_pretrained_backbone(bottom_up)
    dummy_data = torch.zeros((1,3,cfg.general.input_shape[0], cfg.general.input_shape[1]), dtype = torch.float32)
    fpn = FPN(bottom_up(dummy_data), use_dwise_conv=True)
    backbone = Backbone(bottom_up, fpn)
    
    rpn_head = RPN_Head(4, len(cfg.anchor_generator.anchor_ratios))
    anchor_gen = Anchor_Generator(cfg.anchor_generator.anchor_sizes, 
                                    cfg.anchor_generator.anchor_ratios, 
                                    cfg.anchor_generator.anchor_strides)
       
    rpn = RPN(head=rpn_head,
            anchor_generator = anchor_gen,
            anchor_matcher = Matcher(cfg.rpn.matcher.thresholds,
                                    cfg.rpn.matcher.labels,
                                    cfg.rpn.matcher.allow_low_quality_matches),
            batch_size_per_image = cfg.rpn.batch_size_per_img,
            positive_fraction = cfg.rpn.positive_fraction,
            pre_nms_topk = cfg.rpn.pre_nms_topk,
            post_nms_topk = cfg.rpn.post_nms_topk,
            nms_thresh = cfg.rpn.nms_thresh,
            loss_weight = cfg.rpn.loss_weights,
            min_box_size = cfg.rpn.min_box_size,
            box2box_transform = Box2BoxTransform(cfg.rpn.box_transform_weights), #equal scaling factors for all level
            box_reg_loss_type = cfg.rpn.box_reg_loss_type,
            )
    
    # load_rpn_weights(rpn)
    
    box_pooler = ROI_Pooler(output_size = cfg.box_pooler.output_shape,
                            scales = cfg.box_pooler.scales,
                            sampling_ratio = cfg.box_pooler.sampling_ratio,
                            pooler_type = cfg.box_pooler.pooler_type,
                            )
    
    box_head = Fast_RCNN_Conv_AP_Head(input_shape = cfg.box_head.input_shape,
                                    conv_dims = cfg.box_head.conv_dims,
                                    fc_dims = cfg.box_head.fc_dims,
                                    conv_norm = cfg.box_head.conv_norm)
    
    box_predictor = Fast_RCNN_Output_Layers(input_shape = cfg.box_predictor.input_shape,
                                            test_score_thresh = cfg.box_predictor.test_score_thresh,
                                            box2box_transform = Box2BoxTransform(cfg.box_predictor.box_transform_weights),
                                            num_classes = cfg.general.num_thing_classes,
                                            box_reg_loss_type = cfg.box_predictor.box_reg_loss_type,
                                            smooth_l1_beta = cfg.box_predictor.smooth_l1_beta,
                                            loss_weight = cfg.box_predictor.loss_weight,
                                            # ignore_value = cfg.general.num_classes-1)
                                            )
    
    mask_pooler = ROI_Pooler(output_size=cfg.mask_pooler.output_size,
                            scales = cfg.mask_pooler.scales,
                            sampling_ratio=cfg.mask_pooler.sampling_ratio,
                            pooler_type=cfg.mask_pooler.pooler_type)
    
    mask_head = Mask_RCNN_Conv_Upsample_Head(input_shape = cfg.mask_head.input_shape,
                                            num_classes = cfg.general.num_thing_classes,
                                            conv_dims = cfg.mask_head.conv_dims,
                                            conv_norm = cfg.mask_head.conv_norm,
                                            loss_weights = cfg.mask_head.loss_weight,
                                            use_dwise_conv = True)
    
    roi_heads = Standard_ROI_Heads( num_classes = cfg.general.num_thing_classes,
                                    batch_size_per_image = cfg.roi_heads.batch_size_per_img,
                                    positive_fraction = cfg.roi_heads.positive_fraction,
                                    proposal_matcher = Matcher(thresholds=cfg.roi_heads.matcher.thresholds,
                                                                labels=cfg.roi_heads.matcher.labels,
                                                                allow_low_quality_matches=cfg.roi_heads.matcher.allow_low_quality_matches),
                                    box_in_features=cfg.roi_heads.box_in_features,
                                    box_pooler=box_pooler,
                                    box_head=box_head,
                                    box_predictor=box_predictor,
                                    mask_in_features=cfg.roi_heads.mask_in_features,
                                    mask_pooler=mask_pooler,
                                    mask_head=mask_head,
                                    train_on_pred_boxes=cfg.roi_heads.train_on_pred_boxes
                                    )
    
    feature_shapes = {
        k: ShapeSpec(channels=256, stride = v) 
        for k, v in zip(["P2", "P3", "P4", "P5"], [4, 8, 16, 32, 64])
        } 
    sem_seg_head = SemSeg_FPN_Head( input_shape = feature_shapes,
                                    num_classes = cfg.general.num_classes,
                                    conv_dims = 128,
                                    common_stride = 4,
                                    loss_weight=cfg.semantic_head.loss_weight,
                                    norm="GN", 
                                    ignore_value = cfg.general.num_classes-1)
    
    pixel_mean = [72.5239, 83.0444, 73.2913] 
    pixel_std = [0.1860, 0.1888, 0.1856] 
    input_format="BGR"
    
    model = PanopticFPN(cfg = cfg,
                        backbone = backbone,
                        proposal_generator = rpn,
                        roi_heads = roi_heads,
                        sem_seg_head = sem_seg_head,
                        combine_overlap_thresh = cfg.panoptic_head.combine_overlap_thresh,
                        combine_stuff_area_thresh = cfg.panoptic_head.combine_stuff_area_thresh,
                        combine_instances_score_thresh = cfg.panoptic_head.combine_instances_score_thresh,
                        pixel_mean = pixel_mean,
                        pixel_std = pixel_std,
                        input_format = input_format)
    
    print(f"Model is initialized with {int(num_params(model))} params. . . . .")
    return model




def num_params(model):
    num_params = 0.0
    for p in list(model.parameters()):
        num_params+=p.view(-1).shape[0]
    return num_params

def map_r50_block(model, pretrained_model):
    for block_id, _ in enumerate(model):
        for layer, pretrain_layer in zip(model[block_id].bottle_neck_conv, pretrained_model[block_id].children()):
            map_weight(layer, pretrain_layer)
        if block_id == 0:
            map_weight(model[block_id].identity_downsample[0], pretrained_model[block_id].downsample[0])
            map_weight(model[block_id].identity_downsample[1], pretrained_model[block_id].downsample[1])

def map_weight(x, pretrain):
    x.weight = pretrain.weight
    
def load_pretrained_backbone(bottom_up):
    pretrained_r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    map_r50_block(bottom_up.res2, pretrained_r50.layer1)
    map_r50_block(bottom_up.res3, pretrained_r50.layer2)
    map_r50_block(bottom_up.res4, pretrained_r50.layer3)
    map_r50_block(bottom_up.res5, pretrained_r50.layer4)
    
    for param in bottom_up.parameters():
        param.requires_grad = False

def load_rpn_weights(rpn):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    pretrained = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    rpn.rpn_head.conv.weight = pretrained.rpn.head.conv[0][0].weight
    rpn.rpn_head.objness_logit_conv.weight = pretrained.rpn.head.cls_logits.weight
    rpn.rpn_head.anchor_delta_conv.weight = pretrained.rpn.head.bbox_pred.weight

if __name__ == "__main__":
    import anyconfig
    import munch
    from Data.data_utils import read_json_file
    import numpy as np
    from Data.augmentation_impl import RandomFlip, ResizeShortestEdge
    from structures.boxes import BoxMode
    cfg = anyconfig.load(r"E:\20231\DATN\Cityscapes-panoptic-segmentation\config.yaml")
    cfg = munch.munchify(cfg)

    model = load_model(cfg)
    from Data.dataset_mapper import Dataset_mapper
    mapper = Dataset_mapper(is_train=True,
                        use_instance_mask = True,
                        augmentations=[RandomFlip()],
                        image_format='BGR',
                        instance_mask_format='polygon',
                        precomputed_proposal_topk=None,
                        recompute_boxes=True)

    annotation = read_json_file('bill_gtFine.json')
    segmentation = []
    for point in annotation['object'][0]['polygon']:
        segmentation.append(point[0])
        segmentation.append(point[1])
    
    data_dict = {
                'file_name': "bill_image.jpg",
                'sem_seg_file_name': "bill_sem_mask.png",
                'annotations': [
                    {
                        'segmentation': [
                            segmentation
                        ],
                        'bbox': [
                            np.array([point[0] for point in annotation['object'][0]['polygon']]).min(),
                            np.array([point[1] for point in annotation['object'][0]['polygon']]).min(),
                            np.array([point[0] for point in annotation['object'][0]['polygon']]).max(),
                            np.array([point[1] for point in annotation['object'][0]['polygon']]).max(),
                        ],
                        # 'iscrowd': 0,
                        'area': 404038,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'category_id': 0,
                    }
                ]
                }
    mapped_data = mapper(data_dict)
    model.train()
    res = model.forward([mapped_data])
    print(res)

