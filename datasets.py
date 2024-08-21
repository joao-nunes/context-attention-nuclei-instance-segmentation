from PIL import Image
from torch.utils.data import Dataset
from generate_bbox import get_bbox
import os
import numpy as np
import torch
import torch.nn as nn

from skimage import feature
from skimage import morphology
from skimage.exposure import match_histograms


class LizardMaskRCNN(Dataset):

    def __init__(self,
                 file_ids,
                 root_dir,
                 match_hist=False,
                 p=0.5,
                 label=2,
                 footprint=None,
                 transforms=None
                 ):

        self.root_dir = root_dir
        self.file_ids = file_ids
        self.transforms = transforms
        if match_hist:
            self.match_histograms = MatchHistograms(
                label=label,
                footprint=footprint)

        self.match_hist = match_hist
        self.p = p

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root_dir,
                                        self.file_ids[idx], "image.png"))

        labels = np.load(os.path.join(
            self.root_dir,
            self.file_ids[idx],
            "classification_map.npy")).astype(np.int64)

        if self.match_hist and np.random.rand() <= self.p:
            image = self.match_histograms(image, labels)

        labels = torch.Tensor(labels)

        instance_seg_map = torch.Tensor(

            np.load(os.path.join(self.root_dir, self.file_ids[idx],
                                 "instance_seg_map.npy")
                    ).astype(np.int32)
        )

        masks, boxes, num_objs, _ = get_bbox(instance_seg_map)
        if instance_seg_map.any() != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor(1)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        if torch.numel(masks) == 0:
            target["labels"] = torch.tensor([0]).to(torch.int64)
        else:
            target["labels"] = torch.tensor([np.unique(
                                             labels * masks[i, :, :])[1]
                                             for i in range(num_objs)]
                                            ).to(torch.int64)
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        target["area"] = area.to(torch.float32)
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.file_ids)



