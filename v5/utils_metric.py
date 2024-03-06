import torch
import torch.nn as nn
import config
import numpy as np
from utils import (
    iou,
    convert_cells_to_bboxes,
    stable_divide,
    nms,
)

def find_matching_target(pred_box, targets):
    for target_box in targets:
        box1 = pred_box
        box2 = target_box
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2, box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2, box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

        iou_score = iou(b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2)
        if iou_score > config.enough_overlap_threshold:
            return target_box
    
    return None

def calculate_precision_recall(predictions, targets):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_box in predictions:
        matching_target = find_matching_target(pred_box, targets)

        if matching_target is not None:
            true_positives += 1
            targets.remove(matching_target)
        else:
            false_positives += 1

    false_negatives = len(targets)

    precision = stable_divide(true_positives, true_positives + false_positives)
    recall = stable_divide(true_positives, true_positives + false_negatives)

    return precision, recall