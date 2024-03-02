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

    num_predictions = len([1 for prediction_batch in decoded_predictions for prediction in prediction_batch if prediction[0] > Config.valid_prediction_threshold])
    num_targets = len([1 for target_batch in decoded_targets for target in target_batch if target[0] > Config.valid_prediction_threshold])

    # same class and ground truth labeled as positive objectiveness
    potential_TP = [(pred[5] == target[5]) and (target[0] > Config.valid_prediction_threshold) for pred, target in zip(decoded_predictions, decoded_targets)]

    ious = ciou([pred[1:5] for pred, is_tp in zip(decoded_predictions, potential_TP) if is_tp],
                [target[1:5] for target, is_tp in zip(decoded_targets, potential_TP) if is_tp],
                is_pred=False)


    true_positives = torch.sum(ious > Config.enough_overlap_threshold).item()
    false_positives = num_predictions - true_positives
    false_negatives = num_targets - true_positives

    precision = stable_divide(true_positives, true_positives + false_positives)
    recall = stable_divide(true_positives, true_positives + false_negatives)
    
    return precision, recall
