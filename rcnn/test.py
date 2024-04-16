import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from PIL import Image
import cv2
import albumentations as A 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
from tqdm import tqdm 
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


if __name__ == '__main__':
    base_dir = os.getcwd()
    print(base_dir)
    print()

    # dataset_path = "Aquarium Combined/"
    dataset_path = "soybean/"
    coco = COCO(os.path.join(base_dir, "rcnn_aqua", dataset_path, "train", "_annotations.coco.json"))
    categories = coco.cats
    n_classes = len(categories.keys())
    classes = [i[1]['name'] for i in categories.items()]

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    model.load_state_dict(torch.load('soy_res50.pt'))
    
    device = torch.device("cuda") 
    model.to(device)
    model.eval()

    torch.cuda.empty_cache()
    from CustomDataset import CustomDataset
    test_dataset = CustomDataset(root=os.path.join(base_dir, "rcnn_aqua", dataset_path), split="test", transforms=get_transforms(False))

    img, _ = test_dataset[24]

    # img = Image.open("test.jpg")
    # transform = torchvision.transforms.ToTensor()
    # img = transform(img)

    img_int = torch.tensor(img*255, dtype=torch.uint8)

    with torch.no_grad():
        prediction = model([img.to(device)])
        pred = prediction[0]

    fig = plt.figure(figsize=(14, 10))
    plt.imshow(draw_bounding_boxes(img_int,
        pred['boxes'][pred['scores'] > 0.3],
        [classes[i] for i in pred['labels'][pred['scores'] > 0.3].tolist()], width=2, colors=(255,0,0)
    ).permute(1, 2, 0))
    plt.savefig('bounding_boxes_output_1.png')
    plt.show()

    def evaluate_model(model, test_dataset, device, threshold=0.5, iou_threshold=0.5):
        model.eval()
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for image, targets in test_dataset:
                image = image.unsqueeze(0).to(device)
                prediction = model(image)[0]
                pred_boxes = prediction['boxes']
                pred_scores = prediction['scores']
                pred_labels = prediction['labels']

                # Apply threshold to filter out low-confidence predictions
                keep = pred_scores > threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                # Get ground truth boxes and labels
                gt_boxes = targets['boxes'].to(device)
                gt_labels = targets['labels'].to(device)

                # Calculate IoU (Intersection over Union) between predicted and ground truth boxes
                iou = torchvision.ops.box_iou(pred_boxes, gt_boxes)

                # Match predicted boxes with ground truth boxes
                matched_gt_indices = (-iou).argsort(dim=1)[:, 0]
                matched_gt_iou = iou[torch.arange(len(pred_boxes)), matched_gt_indices]

                # Count true positives, false positives, and false negatives
                for i in range(len(pred_boxes)):
                    if matched_gt_iou[i] >= iou_threshold and pred_labels[i] == gt_labels[matched_gt_indices[i]]:
                        true_positives += 1
                    else:
                        false_positives += 1

                false_negatives += len(gt_boxes) - len(pred_boxes)

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)

        print(true_positives)
        print(false_positives)
        print(false_negatives)
        print(precision)
        print(recall)


        return precision, recall
   
    precision, recall = evaluate_model(model, test_dataset, device)