import torch
import torch.nn as nn
from config import Config
import numpy as np
from utils import (
    ciou,
    decodePrediction_bbox_no_offset,
)

def calculate_precision_recall(predictions, targets, scaled_anchor):
    obj = targets[..., 0] == 1

    num_obj = len(targets[-1] == 1)
    num_pred_obj = len(predictions[-1] == 1)

    # Reshaping anchors to match predictions
    scaled_anchor = scaled_anchor.reshape(1, 3, 1, 1, 2)
    
    box_preds = decodePrediction_bbox_no_offset(predictions, scaled_anchor)
    cious = ciou(box_preds[obj], targets[..., 1:5][obj])

     # Filter predictions based on CIoU threshold
    true_positives = torch.sum(cious > Config.enough_overlap_threshold*1.1).item()
    false_positives = num_pred_obj - true_positives

    # targets[..., 1:5][obj].shape[0] is number of bounding boxes
    false_negatives = num_obj - true_positives

    precision = true_positives / (true_positives + false_positives + Config.numerical_stability)
    recall = true_positives / (true_positives + false_negatives + Config.numerical_stability)

    return precision, recall