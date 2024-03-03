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

    num_predictions = 0
    num_targets = 0
    true_positives = 0

    for batch_idx in range(Config.test_batch_size):
        decoded_predictions_batch = decoded_predictions[batch_idx]
        decoded_targets_batch = decoded_targets[batch_idx]

        decoded_predictions_nms = torch.tensor(nms(decoded_predictions_batch.tolist())).view(1, -1, 6)
        num_predictions += torch.sum(decoded_predictions_nms[:, :, 0] > Config.valid_prediction_threshold).item()
        num_targets += torch.sum(decoded_targets_batch[:, 0] > Config.valid_prediction_threshold).item()

        tar_obj = decoded_targets_batch[..., 0] > Config.valid_prediction_threshold
        pred_obj = decoded_predictions_batch[..., 0] > Config.valid_prediction_threshold

        ious = ciou(decoded_predictions_batch[..., 1:5][pred_obj & tar_obj], decoded_targets_batch[..., 1:5][pred_obj & tar_obj], is_pred=False)

        true_positives += torch.sum(ious > Config.enough_overlap_threshold).item()

    return true_positives, num_predictions, num_targets