import torch
import torch.nn as nn
import numpy as np
from utils import (
    ciou,  
)

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        # Identifying object and no-object cells in target
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce(
            pred[..., 0:1][no_obj], target[..., 0:1][no_obj],
        )

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # Adjusting predictions for box coordinates
        box_preds = torch.cat([
            self.sigmoid(pred[..., 1:3]),  # Center x, y with sigmoid activation
            torch.exp(pred[..., 3:5]) * anchors  # Width and height
        ], dim=-1)
        
        cious = ciou(box_preds[obj], target[..., 1:5][obj])

        # CIoU loss for bounding box regression
        box_loss = torch.mean(1-cious)

        # Objectness loss for predicting the presence of an object
        object_loss = self.bce(self.sigmoid(pred[..., 0:1][obj]), target[..., 0:1][obj])

        # Calculating class loss
        class_loss = self.cross_entropy(pred[..., 5:][obj], target[..., 5][obj].long())

        return ( 
			2 * box_loss 
			+ object_loss 
			+ no_object_loss 
			+ class_loss 
		)
