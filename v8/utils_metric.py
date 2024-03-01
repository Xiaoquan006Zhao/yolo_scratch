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

    potential_TP =  (targets[..., 0] > 0.8)

    num_predictions = len(predictions[-1] > 0.8)
    num_targets = len(targets[-1] > 0.8)

    ious = ciou(predictions[..., 1:5][potential_TP], targets[..., 1:5][potential_TP], is_pred=False)

    true_positives = torch.sum(ious > Config.enough_overlap_threshold).item()
    false_positives = num_predictions - true_positives
    false_negatives = num_targets - true_positives


    precision = true_positives / (true_positives + false_positives + Config.numerical_stability)
    recall = true_positives / (true_positives + false_negatives + Config.numerical_stability)
    
    return precision, recall
