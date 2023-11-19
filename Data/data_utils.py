import json
import numpy as np
import torch
from .class_list import LABEL_DICT

import sys
# sys.path.append(r"E:\20231\DATN\detectron2_rebuild")
from structures import Boxes, Instances

def read_json_file(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def write_json_file(path, data):
    with open(path, "w") as json_file:
        json_obj = json.dumps(data, indent=1)
        json_file.write(json_obj)


def annotations_to_instances(annotation_path):
    annotations = read_json_file(annotation_path)
    gt_instances = Instances(
        image_size=(annotations["imgHeight"], annotations["imgWidth"])
    )
    gt_labels = []
    gt_boxes = []
    for obj in annotations["objects"]:
        polygon = np.array(obj["polygon"])
        x1 = polygon[:, 0].min()
        x2 = polygon[:, 0].max()
        y1 = polygon[:, 1].min()
        y2 = polygon[:, 1].max()
        gt_boxes.append(torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
        gt_labels.append(LABEL_DICT[obj["label"]]["id"])
    gt_boxes = torch.cat(gt_boxes)
    gt_instances.gt_boxes = Boxes(gt_boxes)
    gt_instances.gt_classes = torch.tensor(gt_labels)
    return gt_instances
