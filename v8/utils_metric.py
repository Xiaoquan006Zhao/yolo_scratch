import torch
import torch.nn as nn
from config import Config
import numpy as np
from utils import (
    ciou,
    decodePrediction,
    nms,
)

def calculate_precision_recall(predictions, targets, scaled_anchor, s):
    decoded_predictions = decodePrediction(predictions, scaled_anchor, s, to_list=False)
    decoded_targets = targets.reshape(decoded_predictions.shape)

    decoded_predictions_flat = decoded_predictions.view(-1, decoded_predictions.shape[-1])
    decoded_targets_flat = decoded_targets.view(-1, decoded_targets.shape[-1])

    decoded_predictions_nms_flat = torch.tensor(nms(decoded_predictions_flat.tolist())).view(decoded_predictions.shape[0], -1, 6)

    num_predictions = torch.sum(decoded_predictions_nms_flat[:, :, 0] > Config.valid_prediction_threshold).item()
    num_targets = torch.sum(decoded_targets_flat[:, 0] > Config.valid_prediction_threshold).item()

    tar_obj = decoded_targets_flat[:, 0] > Config.valid_prediction_threshold
    pred_obj = decoded_predictions_flat[:, 0] > Config.valid_prediction_threshold

    ious = ciou(decoded_predictions_flat[:, 1:5][pred_obj & tar_obj], decoded_targets_flat[:, 1:5][pred_obj & tar_obj], is_pred=False)

    true_positives = torch.sum(ious > Config.enough_overlap_threshold).item()

    return true_positives, num_predictions, num_targets
