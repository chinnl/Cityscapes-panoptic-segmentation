from .dataset_mapper import Dataset_mapper
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .data_utils import read_json_file
from .labels import name2label
from structures import BoxMode
import os
import glob
from .augmentation_impl import RandomFlip, ResizeShortestEdge
from .distributed_sampler import TrainingSampler, InferenceSampler
from .common import ToIterableDataset, MapDataset


class Cityscapes(Dataset):
    def __init__(self, 
                 data_folder: str, #"D:\Data\Cityscapes\leftImg8bit\train",
                 ):
        super().__init__()
        self.data_folder = data_folder
        image_list = glob.glob(data_folder + "\\*\\*")
        gt_dir = data_folder.replace("\\leftImg8bit\\", "\\gtFine\\")
        
        self.file_name = []
        self.sem_seg_file_name = []
        self.annotations = []
        
        for idx in range(len(image_list)):
            file_name = image_list[idx].split("\\")[-1].replace("_leftImg8bit.png", "")
            city_name = file_name.split("_")[0]
            annotation = read_json_file(os.path.join(gt_dir, city_name + "\\" + file_name + "_gtFine_polygons.json"))
            
            self.file_name.append(image_list[idx])
            self.sem_seg_file_name.append(os.path.join(gt_dir, city_name + "\\" + file_name + "_gtFine_labelIds.png"))

            anno = []
            
            for obj_dict in annotation['objects']:
                iscrowd = int(obj_dict['label'].endswith('group'))
                
                polygon = np.array(obj_dict['polygon'])
                bbox = obj_dict['polygon'][np.argmin(np.sum(polygon, axis = 1))] + obj_dict['polygon'][np.argmax(np.sum(polygon, axis = 1))]
                
                try:
                    category_id = name2label[obj_dict['label']].categoryId
                except:
                    category_id = name2label[obj_dict['label'].replace("group", "")].categoryId
                    
                anno.append(
                    {
                        'segmentation': [polygon],
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'iscrowd': iscrowd,
                        'category_id': category_id,
                    }
                )
            self.annotations.append(anno)
            
            if idx == 15: 
                break
            
    def __len__(self):
        return len(self.file_name)
    
    def __getitem__(self, idx):
        return {
            'file_name': self.file_name[idx],
            'sem_seg_file_name': self.sem_seg_file_name[idx],
            'annotations': self.annotations[idx]
        }
    
    
def build_dataloader(cfg):
    train_folder = cfg.train.folder
    val_folder = cfg.val.folder
    
    data_mapper = Dataset_mapper(is_train=True,
                            use_instance_mask = True,
                            augmentations=[RandomFlip(), 
                                           ResizeShortestEdge([512], sample_style='choice')],
                            image_format='BGR',
                            instance_mask_format='bitmask',
                            precomputed_proposal_topk=None,
                            recompute_boxes=True)
    
    cityscapes_train = MapDataset(Cityscapes(train_folder), data_mapper)
    cityscapes_val = MapDataset(Cityscapes(val_folder), data_mapper)

    train_sampler = TrainingSampler(len(cityscapes_train))
    val_sampler = TrainingSampler(len(cityscapes_val))
    
    cityscapes_train = ToIterableDataset(cityscapes_train, 
                                         train_sampler, 
                                         shard_chunk_size=cfg.train.batch_size)
    cityscapes_val = ToIterableDataset(cityscapes_val, 
                                         val_sampler, 
                                         shard_chunk_size=cfg.val.batch_size)
    
    trainloader = DataLoader(cityscapes_train,
                            batch_size=cfg.train.batch_size,
                            drop_last=False,
                            num_workers=4,
                            collate_fn=trivial_batch_collator,
                            prefetch_factor = None,
                            persistent_workers = False,
                            pin_memory = False
                            )
    
    valloader = DataLoader(cityscapes_val,
                            batch_size=cfg.val.batch_size,
                            drop_last=False,
                            num_workers=4,
                            collate_fn=trivial_batch_collator,
                            prefetch_factor = None,
                            persistent_workers = False,
                            pin_memory = False
                            )
    
    return trainloader, valloader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch
