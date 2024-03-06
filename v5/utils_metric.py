import torch
import torch.nn as nn
import config
import numpy as np
from utils import (
    iou,
    stable_divide,
)

def calculate_precision_recall(predictions, targets):
    # Convert the predictions and targets lists to NumPy arrays for efficient calculations
    predictions_np = np.array(predictions)
    targets_np = np.array(targets)

    # Extract bounding box coordinates for ease of computation
    pred_boxes = predictions_np[:, :4]
    target_boxes = targets_np[:, :4]

    # Calculate IoU matrix using NumPy broadcasting
    intersection_areas = np.maximum(0, np.minimum(pred_boxes[:, 2:], target_boxes[:, 2:]) - np.maximum(pred_boxes[:, :2], target_boxes[:, :2]))
    intersection_areas = np.prod(intersection_areas, axis=1)
    
    pred_areas = np.prod(pred_boxes[:, 2:] - pred_boxes[:, :2], axis=1)
    target_areas = np.prod(target_boxes[:, 2:] - target_boxes[:, :2], axis=1)

    union_areas = pred_areas + target_areas - intersection_areas

    iou = intersection_areas / np.maximum(union_areas, 1e-8)  # Avoid division by zero

    # Initialize variables to track true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through each predicted bounding box
    for i in range(len(predictions)):
        # Find the index of the maximum IoU in the corresponding row
        matching_target_index = np.argmax(iou[i])

        # If IoU is above the threshold, consider it a match
        if iou[i, matching_target_index] > 0.5:
            true_positives += 1
            # Remove the matching target index to avoid double counting
            iou[:, matching_target_index] = 0
        else:
            false_positives += 1

    # The remaining unmatched targets are false negatives
    false_negatives = len(targets) - true_positives

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

# Example usage
predictions = [(10, 10, 20, 20, 1), (30, 30, 15, 15, 2)]
targets = [(12, 12, 18, 18, 1), (35, 35, 10, 10, 2)]

precision, recall = calculate_precision_recall(predictions, targets)
print(f'Precision: {precision}, Recall: {recall}')
