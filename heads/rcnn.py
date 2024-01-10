import numpy as np
import torch
from torch import nn
from typing import List, Dict, Optional, Tuple

from utils.postprocess import detector_postprocess
from layers import move_device_like
from structures import ImageList


class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feat nure extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """
    def __init__(
        self,
        *,
        backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        **kwargs
        ):
        super().__init__()
        self.cfg = kwargs['cfg']
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.input_format = input_format
        self.pixel_mean = torch.tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(-1, 1, 1)
        
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"Pixel mean ({self.pixel_mean}) and pixel std ({self.pixel_std}) have different shapes!"
    
    @property
    def device(self):
        return self.pixel_mean.device
    
    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
    
    def forward(self, batched_inputs: List[Dict[str, torch.tensor]]):
        '''
        Args:
            batched_inputs:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
        Return:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks"
        '''
        if not self.training:
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        features = self.backbone(images.tensor)
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0], "Input must have proposals in case proposal generator is not defined"
            proposals = [x['proposals'].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    
    def inference(self, 
                  batched_inputs: List[Dict[str, torch.tensor]],
                  detected_instances: Optional[List[ImageList]] = None,
                  do_postprocess:bool = True):
        
        assert not self.training, "The model is in `training` mode!"
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0], "Input must have proposals in case proposal generator is not defined"
                proposals = [x['proposals'].to(self.device) for x in batched_inputs]
            
            results, _ = self.roi_heads(images, features, proposals, None)
        
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
        if do_postprocess:
            return self.post_process(results, batched_inputs, images.image_sizes)
        return results
    
    def preprocess_image(self, batched_inputs):
        images = [self._move_to_current_device(x['images']) for x in batched_inputs]
        images = [(x - self.pixel_mean)/self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.cfg.general.padding_constraint.size_divisibility,
            padding_constraints=self.cfg.general.padding_constraint,
        )
        return images
    
    def post_process(self, instances, batched_inputs, image_sizes):
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(instances, batched_inputs, image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({'instances': r})
        return processed_results
        