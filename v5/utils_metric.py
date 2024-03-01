import torch
import torch.nn as nn
from config import Config
import numpy as np
from utils import (
    ciou,
    decodePrediction,
)

def calculate_precision_recall(predictions, targets, scaled_anchor, s):
    predictions = decodePrediction(predictions, scaled_anchor, s, to_list=False)

    # objectiveness stored at 1
    obj = targets[..., 0] == 1
    print(targets[...,:][obj][0])
    potential_TP =  (targets[..., 1] > Config.valid_prediction_threshold)

    num_predictions = len(predictions[-1][1] > Config.valid_prediction_threshold)
    num_targets = len(targets[-1][1] > Config.valid_prediction_threshold)

    ious = ciou(predictions[..., 1:5][potential_TP], targets[..., 1:5][potential_TP], is_pred=False)

    true_positives = torch.sum(ious > Config.enough_overlap_threshold).item()
    false_positives = num_predictions - true_positives
    false_negatives = num_targets - true_positives

    precision = true_positives / (true_positives + false_positives + Config.numerical_stability)
    recall = true_positives / (true_positives + false_negatives + Config.numerical_stability)
    
    return precision, recall
