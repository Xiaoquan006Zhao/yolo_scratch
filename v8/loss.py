import torch
import torch.nn as nn
import math
from utils import (
    ciou,  
    decodePrediction_bbox_no_offset,
)

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, scaled_anchor, scale):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_object_loss = self.bce(
            pred[..., 0:1][no_obj], target[..., 0:1][no_obj],
        )

        scaled_anchor = scaled_anchor.reshape(1, 3, 1, 1, 2)
        box_preds = decodePrediction_bbox_no_offset(pred, scaled_anchor)
        cious = ciou(box_preds[obj], target[..., 1:5][obj])
        box_loss = torch.mean(1-cious)
        object_loss = self.bce(self.sigmoid(pred[..., 0:1][obj]), target[..., 0:1][obj])
        class_loss = self.cross_entropy(pred[..., 5:][obj], target[..., 5][obj].long())

        loss = box_loss + object_loss + no_object_loss + class_loss 
        # assert not math.isnan(loss), f"{box_loss}, {object_loss}, {no_object_loss}, {class_loss}, {cious}, \n {box_preds[obj]}, \n {target[..., 1:5][obj]}"

        return (loss)
