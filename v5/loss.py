import torch
import torch.nn as nn
import math
import numpy as np
from utils import (
    ciou,  
    decodePrediction_bbox,
)

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, scaled_anchor, scale):
        # Identifying object and no-object cells in target
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce(
            pred[..., 0:1][no_obj], target[..., 0:1][no_obj],
        )

        # Reshaping anchors to match predictions
        scaled_anchor = scaled_anchor.reshape(1, 3, 1, 1, 2)

        box_preds = decodePrediction_bbox(pred, scaled_anchor, scale)
        box_targets = decodePrediction_bbox(target, scaled_anchor, scale)
        
        cious = ciou(box_preds[obj], box_targets[obj])

        # CIoU loss for bounding box regression
        box_loss = torch.mean(1-cious)

        # Objectness loss for predicting the presence of an object
        object_loss = self.bce(self.sigmoid(pred[..., 0:1][obj]), target[..., 0:1][obj])

        # Calculating class loss
        class_loss = self.cross_entropy(pred[..., 5:][obj], target[..., 5][obj].long())

        loss = 2 * box_loss + object_loss + no_object_loss + class_loss 
        assert not math.isnan(loss), f"{box_loss}, {object_loss}, {no_object_loss}, {class_loss}, {cious}, \n {box_preds[obj]}, \n {target[..., 1:5][obj]}"

        return ( 
            box_loss 
            + object_loss 
            + no_object_loss 
            + class_loss 
        )
