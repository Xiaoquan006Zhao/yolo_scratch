import torch
import torch.nn as nn
from config import Config
from utils import (
    ciou,  
    decodePrediction_bbox_no_offset,
)

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sigmoid = nn.Sigmoid()

        # self.weight_box = nn.Parameter(torch.ones(1)).to(Config.device)
        # self.weight_object = nn.Parameter(torch.ones(1)).to(Config.device)
        # self.weight_no_object = nn.Parameter(torch.ones(1)).to(Config.device)
        # self.weight_class = nn.Parameter(torch.ones(1)).to(Config.device)

    def forward(self, pred, target, scaled_anchor, scale):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_object_loss = self.bce(
            pred[..., 0:1][no_obj], target[..., 0:1][no_obj],
        )

        scaled_anchor = scaled_anchor.reshape(1, 3, 1, 1, 2)
        box_preds = decodePrediction_bbox_no_offset(pred, scaled_anchor)
        # cious = (box_preds[obj], target[..., 1:5][obj])
        box_loss = self.mse(box_preds[obj], target[..., 1:5][obj])

        object_loss = self.bce(pred[..., 0:1][obj], target[..., 0:1][obj])
        class_loss = self.cross_entropy(pred[..., 5:][obj], target[..., 5][obj].long())

        # weighted_box_loss = self.weight_box * box_loss.to(Config.device)
        # weighted_object_loss = self.weight_object * object_loss.to(Config.device)
        # weighted_no_object_loss = self.weight_no_object * no_object_loss.to(Config.device)
        # weighted_class_loss = self.weight_class * class_loss.to(Config.device)

        # loss = weighted_box_loss + weighted_object_loss + weighted_no_object_loss + weighted_class_loss

        loss = box_loss + object_loss + no_object_loss + class_loss


        return loss