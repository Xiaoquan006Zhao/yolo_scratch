import torch
import torch.nn as nn
import config
import numpy as np
from utils import (
    ciou,
    convert_cells_to_bboxes,
    stable_divide
)

def calculate_precision_recall(predictions, targets, scaled_anchor, grid_size):
    decoded_predictions = convert_cells_to_bboxes(predictions, scaled_anchor, grid_size, to_list=False)
    decoded_targets = targets.reshape(decoded_predictions.shape)

    num_predictions = torch.sum(decoded_predictions[:, :, 0] >  config.valid_prediction_threshold).item()
    num_targets = torch.sum(decoded_targets[:, :, 0] >  config.valid_prediction_threshold).item()

    tar_obj = decoded_targets[..., 0] > config.valid_prediction_threshold
    pred_obj = decoded_predictions[..., 0] > config.valid_prediction_threshold

    # IoU to calculate the overlap between prediction and ground truth
    ious = ciou(decoded_predictions[..., 1:5][pred_obj & tar_obj], decoded_targets[..., 1:5][pred_obj & tar_obj], mode=config.CIOU_MODE.IoU)

    true_positives = torch.sum(ious > config.enough_overlap_threshold).item()
    
    precision = stable_divide(true_positives, num_predictions)
    recall = stable_divide(true_positives, num_targets)
    
    return precision, recall