import torch
import torch.nn as nn
import config
import numpy as np
from utils import (
    stable_divide,
)

def find_matching_target(pred_box, targets):
    for target_box in targets:
        if pred_box[5] == target_box[5]:
            box1 = pred_box[1:5]
            box2 = target_box[1:5]
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

            area_box1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            area_box2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

            intersection_x = [max(b1_x1, b2_x1), min(b1_x2, b2_x2)]
            intersection_y = [max(b1_y1, b2_y1), min(b1_y2, b2_y2)]
            
            area_intersection = (intersection_x[1] - intersection_x[0]) * (intersection_y[1] - intersection_y[0])

            iou_score = area_intersection / (area_box1 + area_box2 - area_intersection)

            if iou_score > config.enough_overlap_threshold:
                targets.remove(target_box)
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
        else:
            false_positives += 1

    false_negatives = len(targets)

    precision = stable_divide(true_positives, true_positives + false_positives)
    recall = stable_divide(true_positives, true_positives + false_negatives)

    return precision, recall