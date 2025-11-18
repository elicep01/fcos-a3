import os
import random

import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .transforms import Compose, ConvertAnnotations, ConvertAnnotationsCOCO, RandomHorizontalFlip, ToTensor


def trivial_batch_collator(batch):
    """
    A batch collator that allows us to bypass auto batching
    """
    return tuple(zip(*batch))


def worker_init_reset_seed(worker_id):
    """
    Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class VOCDetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load VOC data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
  

    def get_cls_names(self):
        cls_names = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        return cls_names

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class COCODetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load COCO data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        sorted_cat_ids = sorted(self.coco.getCatIds())
        self.lookup_table = [-1 for i in range(100)]
        for i in range(80):
          self.lookup_table[sorted_cat_ids[i]] = i
        self.lookup_table = torch.tensor(self.lookup_table, dtype = torch.int64)
        
    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        # for annotation in annotations:
        #   annotation["category_id"] = self.lookup_table[annotation["category_id"]]
          
        labels = torch.tensor([annotation["category_id"] for annotation in annotations], dtype = torch.int64)
        mapped_labels = self.lookup_table[labels]


        for annotation, mapped_label in zip(annotations, mapped_labels):
          annotation["category_id"] = int(mapped_label)
        target = dict(image_id=image_id, annotations=annotations)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

def build_dataset(name, split, img_folder, json_folder):
    """
    Create VOC dataset with default transforms for training / inference.
    New datasets can be linked here.
    """
    if name == "VOC2007":
        assert split in ["trainval", "test"]
        is_training = split == "trainval"
        if is_training:
            transforms = Compose([ConvertAnnotations(), RandomHorizontalFlip(), ToTensor()])
        else:
            transforms = Compose([ConvertAnnotations(), ToTensor()])

    elif name == "COCO":
        assert split in ["instances_train2017", "instances_val2017"]
        is_training = split == "instances_train2017"
        if is_training:
            transforms = Compose([ConvertAnnotationsCOCO(), RandomHorizontalFlip(), ToTensor()])
        else:
            transforms = Compose([ConvertAnnotationsCOCO(), ToTensor()])
    else: 
        print("Unsupported dataset")
        return None

    if name == "VOC2007":
        dataset = VOCDetection(
            img_folder, os.path.join(json_folder, split + ".json"), transforms
        )
    elif name == "COCO":
        dataset = COCODetection(
            img_folder, os.path.join(json_folder, split + ".json"), transforms
        )      

    return dataset


def build_dataloader(dataset, is_training, batch_size, num_workers):
    """
    Create a dataloder for VOC dataset
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        persistent_workers=True,
    )
    return loader
