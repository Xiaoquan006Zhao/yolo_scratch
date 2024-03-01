import torch
import torch.nn as nn
from config import Config
import numpy as np
from utils import (
    ciou,
    decodePrediction,
)

def calculate_precision_recall(predictions, targets, scaled_anchor, s):
    # Decode predictions
    predictions = decodePrediction(predictions, scaled_anchor, s, to_list=False)
    
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for batch in range(predictions.shape[0]):
        for anchor in range(predictions.shape[1]):
            for row in range(predictions.shape[2]):
                for col in range(predictions.shape[3]):
                    pred_bbox = predictions[batch, anchor, row, col, 1:5]
                    target_bbox = targets[batch, anchor, row, col, 1:5]  
                    pred_obj, target_obj = predictions[batch, anchor, row, col, 0], targets[batch, anchor, row, col, 0]
                    pred_class, target_class = predictions[batch, anchor, row, col, 5], targets[batch, anchor, row, col, 5]

                    ciou = ciou(pred_bbox, target_bbox, is_pred=False)
                    
                    # Check confidence and class alignment
                    if pred_obj > 0.8:
                        if pred_class == target_class and ciou > Config.enough_overlap_threshold:
                            true_positives += 1
                        else:
                            false_positives += 1
                    elif target_obj > 0.8:
                        false_negatives += 1
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return precision, recall
