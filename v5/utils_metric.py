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
    decoded_predictions = decodePrediction(predictions, scaled_anchor, s, to_list=False)
    decoded_targets = targets.reshape(decoded_predictions.shape)

    num_predictions = torch.sum(decoded_predictions[:, :, 0] >  Config.valid_prediction_threshold).item()
    num_targets = torch.sum(decoded_targets[:, :, 0] >  Config.valid_prediction_threshold).item()

    tar_obj = decoded_targets[..., 0] > Config.valid_prediction_threshold
    pred_obj = decoded_predictions[..., 0] > Config.valid_prediction_threshold

    ious = ciou(predictions[..., 1:5][pred_obj], targets[..., 1:5][tar_obj], is_pred=False)

    true_positives = torch.sum(ious > Config.enough_overlap_threshold).item()
    false_positives = num_predictions - true_positives
    false_negatives = num_targets - true_positives

    precision = stable_divide(true_positives, true_positives + false_positives)
    recall = stable_divide(true_positives, true_positives + false_negatives)
    
    return precision, recall
