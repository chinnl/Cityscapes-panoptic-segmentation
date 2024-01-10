import numpy as np
import torch
from typing import List, Union, Optional
import copy
from fvcore.transforms.transform import Transform
from .data_utils import *
from .augmentation import Augmentation, AugmentationList, AugInput


class Dataset_mapper:
    def __init__(self,
                 is_train: bool,
                 *, 
                 augmentations: List[Union[Augmentation, Transform]],
                 image_format: str, #PIL, or "BGR" or "YUV-BT.601"
                 use_instance_mask: bool = False, #whether to process instance segmentation annotations, if available
                 instance_mask_format:str = "polygon", #one of "polygon" or "bitmask". Process instance segmentation
                                                        #masks into this format.
                 precomputed_proposal_topk: Optional[int] = None, #if given, will load pre-computed proposals from dataset_dict 
                                                                # and keep the top k proposals for each image.
                 recompute_boxes: bool = False, #whether to overwrite bounding box annotations
                                                # by computing tight bounding boxes from instance mask annotations.
                 ):
        if recompute_boxes:
            assert use_instance_mask, "Recompute boxes require instance masks!"
            
        self.is_train = is_train
        if augmentations is not None:
            self.augmentations = AugmentationList(augmentations)
        else:
            self.augmentations = None
         
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.proposal_topk = precomputed_proposal_topk
        self.recompute_boxes = recompute_boxes
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        image = read_image(dataset_dict['file_name'], format = self.image_format)
        check_image_size(dataset_dict, image)
        
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        
        if self.augmentations is not None:
            aug_input = AugInput(image, sem_seg= sem_seg_gt)
            transforms = self.augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        else:
            transforms = None
        
        image_shape = image.shape[:2]
        
        dataset_dict["images"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1)))
        if sem_seg_gt is not None:
            dataset_dict['sem_seg'] = torch.tensor(sem_seg_gt.copy()).to(torch.long)
        
        if self.proposal_topk is not None:
            transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk = self.proposal_topk
            )
        
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        
        if "annotations" in dataset_dict and transforms is not None:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        for anno in dataset_dict['annotations']:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
        annos = [
            transform_instance_annotations(
                obj, transforms, image_shape
            ) for obj in dataset_dict.pop('annotations')
            if obj.get('iscrowd', 0) == 0
        ]
        instances = annotations_to_instances(
            annos, image_shape, mask_format = self.instance_mask_format
        )
        
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict['instances'] = filter_empty_instances(instances)
    
if __name__ == '__main__':
    from Data.data_utils import read_json_file
    from Data.augmentation_impl import RandomFlip, ResizeShortestEdge
    from structures.boxes import BoxMode
    mapper = Dataset_mapper(is_train=True,
                        use_instance_mask = True,
                        augmentations=[RandomFlip(), 
                                       ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800),
                                                          sample_style="choice",
                                                          max_size=1333,)],
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
                ],
                
                }
    mapped_data = mapper(data_dict)
    print(np.array(mapped_data['instances'].gt_masks[0]).shape)