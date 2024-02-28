import torch
import torch.nn as nn
import math
import numpy as np
from utils import (
    ciou,  
    decodePrediction_bbox,
    decodePrediction_bbox_no_offset,
)

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

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
        
        box_preds = decodePrediction_bbox_no_offset(pred, scaled_anchor)
        
        cious = ciou(box_preds[obj], target[..., 1:5][obj])

        # CIoU loss for bounding box regression
        box_loss = torch.mean(1-cious)

        # Objectness loss for predicting the presence of an object
        object_loss = self.bce(self.sigmoid(pred[..., 0:1][obj]), target[..., 0:1][obj])

        # Calculating class loss
        class_loss = self.cross_entropy(pred[..., 5:][obj], target[..., 5][obj].long())

        loss = box_loss + object_loss + no_object_loss + class_loss 
        assert not math.isnan(loss), f"{box_loss}, {object_loss}, {no_object_loss}, {class_loss}, {cious}, \n {box_preds[obj]}, \n {target[..., 1:5][obj]}"

        return (loss)
