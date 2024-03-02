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
    decoded_predictions = decodePrediction(predictions, scaled_anchor, s)
    decoded_targets = targets.reshape(len(decoded_predictions), len(decoded_predictions[0]), 6)

    # same class and ground truth labeled as positive objectiveness
    potential_TP = [(pred[5] == target[5]) and (target[0] > Config.valid_prediction_threshold) for pred, target in zip(decoded_predictions, decoded_targets)]

    num_predictions = len([1 for prediction in decoded_predictions if prediction[0] > Config.valid_prediction_threshold])
    num_targets = len([1 for target in decoded_targets if target[0] > Config.valid_prediction_threshold])

    ious = ciou(np.array(decoded_predictions)[potential_TP, 1:5], np.array(decoded_targets)[potential_TP, 1:5], is_pred=False)

    true_positives = torch.sum(ious > Config.enough_overlap_threshold).item()
    false_positives = num_predictions - true_positives
    false_negatives = num_targets - true_positives

    precision = stable_divide(true_positives, true_positives + false_positives)
    recall = stable_divide(true_positives, true_positives + false_negatives)
    
    return precision, recall
