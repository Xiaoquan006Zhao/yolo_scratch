import torch
import pandas as pd
import os
import numpy as np
import config
from PIL import Image
from utils import (
    ciou,
)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        label_dir,
        anchors,
        image_size,
        grid_sizes,
        num_classes,
        transform=None,
    ):
        self.label_list = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.grid_sizes = grid_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_classes = num_classes

        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1])
        # We are applying roll to move class label to the last column
        # 5 columns: x, y, width, height, class_label
        bboxes = np.roll(
            np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
        ).tolist()

        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        targets = [torch.zeros((s, s, 5)) for s in self.grid_sizes]

        return image, tuple(targets)
