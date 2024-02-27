import torch
import torch.nn as nn
import config
from utils import (
    ciou,
)


def calculate_precision_recall(predictions, targets, scaled_anchor):
    sigmoid = nn.Sigmoid()
    obj = targets[..., 0] == 1
     # Reshaping anchors to match predictions
    scaled_anchor = scaled_anchor.reshape(1, 3, 1, 1, 2)
    
    # No need to fully decode the prediction, as we don't care about the offset the upper left corner
    # to the assigned grid when calculating ciou
    box_preds = torch.cat([sigmoid(predictions[..., 1:3]), 
                            torch.exp(predictions[..., 3:5]) * scaled_anchor 
                        ],dim=-1) 
    
    cious = ciou(box_preds[obj], targets[..., 1:5][obj])

     # Filter predictions based on CIoU threshold
    true_positives = torch.sum(cious > config.enough_overlap_threshold*1.1).item()
    # targets[..., 1:5][obj].shape[0] is number of bounding boxes
    false_positives = targets[..., 1:5][obj].shape[0] - true_positives
    false_negatives = targets[..., 1:5][obj].shape[0] - true_positives

    precision = true_positives / (true_positives + false_positives + config.numerical_stability)
    recall = true_positives / (true_positives + false_negatives + config.numerical_stability)

    return precision, recall