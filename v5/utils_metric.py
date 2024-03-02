import torch
import torch.nn as nn
from config import Config
import numpy as np
from utils import (
    ciou,
    decodePrediction,
    stable_divide
)

def calculate_precision_recall(predictions, targets, scaled_anchor, s):
    decoded_predictions = decodePrediction(predictions, scaled_anchor, s, batch_seperate=False)
    decoded_targets = targets.reshape(len(decoded_predictions), 6)

    # same class and ground truth labeled as positive objectiveness
    potential_TP =  (decoded_predictions[5] == decoded_targets[5]) & (decoded_targets[0] > Config.valid_prediction_threshold)

    num_predictions = len([1 for prediction in decoded_predictions[0] if prediction > Config.valid_prediction_threshold])
    num_targets = len([1 for target in decoded_targets[0] if target > Config.valid_prediction_threshold])

    ious = ciou(decoded_predictions[1:5][:, potential_TP], decoded_targets[1:5][:, potential_TP], is_pred=False)

    true_positives = torch.sum(ious > Config.enough_overlap_threshold).item()
    false_positives = num_predictions - true_positives
    false_negatives = num_targets - true_positives

    precision = stable_divide(true_positives, true_positives + false_positives)
    recall = stable_divide(true_positives, true_positives + false_negatives)
    
    return precision, recall
