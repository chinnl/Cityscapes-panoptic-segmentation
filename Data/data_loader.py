from .dataset_mapper import Dataset_mapper
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .data_utils import read_json_file
from .labels import name2label, labels
from structures import BoxMode
import os
import glob
from .augmentation_impl import RandomFlip, ResizeShortestEdge
import cv2
# from .distributed_sampler import TrainingSampler
# from .common import ToIterableDataset, MapDataset


class Cityscapes(Dataset):
    def __init__(self, 
                 data_folder: str,
                 ):
        super().__init__()
        num_classes = len(set([label.trainId for label in labels]))
        self.data_folder = data_folder
        self.image_list = glob.glob(data_folder + "/*/*")
        gt_dir = data_folder.replace("/leftImg8bit/", "/gtFine/")
        self.data_mapper = Dataset_mapper(is_train=True,
                            use_instance_mask = True,
                            augmentations=[RandomFlip(), 
                                           ResizeShortestEdge([512], sample_style='choice')],
                            image_format='BGR',
                            instance_mask_format='bitmask',
                            precomputed_proposal_topk=None,
                            recompute_boxes=True,
                            label_mapping={255: num_classes - 1})
                            
        self.file_name = []
        self.sem_seg_file_name = []
        self.annotations = []
        _labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
        dataset_trainId_to_contiguous_id = {l.trainId: idx for idx, l in enumerate(_labels)}
        
        for idx in range(len(self.image_list)):
            file_name = self.image_list[idx].split("/")[-1].replace("_leftImg8bit.png", "")
            city_name = file_name.split("_")[0]
            annotation = read_json_file(os.path.join(gt_dir, city_name + "/" + file_name + "_gtFine_polygons.json"))

            anno = []
            
            for obj_dict in annotation['objects']:
                if "deleted" in obj_dict:  # cityscapes data format specific
                    continue
                iscrowd = int(obj_dict['label'].endswith('group'))
                
                try: 
                    label = name2label[obj_dict['label']]
                except:
                    label = name2label[obj_dict['label'].replace("group", "")]
                    
                if label.id < 0:  # cityscapes data format
                    continue
                
                category_id = label.trainId
                # if category_id == 255: category_id = num_classes - 1
                if not label.hasInstances or label.ignoreInEval:
                    continue
                
                category_id = dataset_trainId_to_contiguous_id[category_id]
                
                polygon = np.array(obj_dict['polygon'])
                bbox = [np.min(polygon[:, 0]), np.min(polygon[:, 1]), np.max(polygon[:, 0]), np.max(polygon[:, 1])]
                
                if cv2.contourArea(polygon) >= 400 and (bbox[2] - bbox[0])*(bbox[3] - bbox[1]) != 0:
                    anno.append(
                        {
                            'segmentation': [polygon],
                            'bbox': bbox,
                            'bbox_mode': BoxMode.XYXY_ABS,
                            'iscrowd': iscrowd,
                            'category_id': category_id,
                        }
                    )
            if len(anno):
                self.annotations.append(anno)
                self.file_name.append(self.image_list[idx])
                self.sem_seg_file_name.append(os.path.join(gt_dir, city_name + "/" + file_name + "_gtFine_labelTrainIds.png"))

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        data_dict = {
            'file_name': self.file_name[idx],
            'sem_seg_file_name': self.sem_seg_file_name[idx],
            'annotations': self.annotations[idx]
        } if len(self.annotations[idx]) else {
            'file_name': self.file_name[idx],
            'sem_seg_file_name': self.sem_seg_file_name[idx],
        }
        return self.data_mapper(data_dict)
    
    
def build_dataloader(cfg):
    train_folder = cfg.train.folder
    val_folder = cfg.val.folder
    
    # data_mapper = Dataset_mapper(is_train=True,
    #                         use_instance_mask = True,
    #                         augmentations=[RandomFlip(), 
    #                                        ResizeShortestEdge([512], sample_style='choice')],
    #                         image_format='BGR',
    #                         instance_mask_format='bitmask',
    #                         precomputed_proposal_topk=None,
    #                         recompute_boxes=True)
    cityscapes_train = Cityscapes(train_folder)
    cityscapes_val = Cityscapes(val_folder)
    
    assert len(cityscapes_train)*len(cityscapes_val) != 0, "Dataset len = 0"
    # cityscapes_train = MapDataset(Cityscapes(train_folder), data_mapper)
    # cityscapes_val = MapDataset(Cityscapes(val_folder), data_mapper)

    # train_sampler = TrainingSampler(len(cityscapes_train))
    # val_sampler = TrainingSampler(len(cityscapes_val))
    
    # cityscapes_train = ToIterableDataset(cityscapes_train, 
    #                                      train_sampler, 
    #                                      shard_chunk_size=cfg.train.batch_size)
    # cityscapes_val = ToIterableDataset(cityscapes_val, 
    #                                      val_sampler, 
    #                                      shard_chunk_size=cfg.val.batch_size)
    
    trainloader = DataLoader(cityscapes_train,
                            batch_size=cfg.train.batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=4,
                            collate_fn=trivial_batch_collator,
                            prefetch_factor = None,
                            persistent_workers = False,
                            pin_memory = True
                            )
    
    valloader = DataLoader(cityscapes_val,
                            batch_size=cfg.val.batch_size,
                            drop_last=False,
                            num_workers=4,
                            collate_fn=trivial_batch_collator,
                            prefetch_factor = None,
                            persistent_workers = False,
                            pin_memory = True
                            )
    
    return trainloader, valloader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch
