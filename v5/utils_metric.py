import torch
import torch.nn as nn
import config
import numpy as np
from utils import (
    ciou,
    convert_cells_to_bboxes,
    stable_divide,
    nms,
)

def find_matching_target(pred_box, targets):
    for target_box in targets:
        iou = ciou(pred_box, target_box, config.CIOU_MODE.IoU)

        if iou > config.enough_overlap_threshold:
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