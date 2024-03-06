import torch
import torch.nn as nn
import config
import numpy as np
from utils import (
    ciou,
    convert_cells_to_bboxes,
    stable_divide
)

def calculate_precision_recall(predictions, targets):
    predictions.reshape(targets.shape)

    num_predictions = torch.sum(predictions[:, :, 0] >  config.valid_prediction_threshold).item()
    num_targets = torch.sum(targets[:, :, 0] >  config.valid_prediction_threshold).item()

    tar_obj = targets[..., 0] > config.valid_prediction_threshold
    pred_obj = predictions[..., 0] > config.valid_prediction_threshold

    # IoU to calculate the overlap between prediction and ground truth
    ious = ciou(predictions[..., 1:5][pred_obj & tar_obj], targets[..., 1:5][pred_obj & tar_obj], mode=config.CIOU_MODE.IoU)

    true_positives = torch.sum(ious > config.enough_overlap_threshold).item()
    
    precision = stable_divide(true_positives, num_predictions)
    recall = stable_divide(true_positives, num_targets)

    return precision, recall