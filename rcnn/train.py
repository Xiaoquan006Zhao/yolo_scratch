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

    from CustomDataset import CustomDataset
    train_dataset = CustomDataset(root=os.path.join(base_dir, "rcnn_aqua", dataset_path), transforms=get_transforms(True))
    
    sample = train_dataset[2]
    img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
    plt.imshow(draw_bounding_boxes(
        img_int, sample[1]['boxes'], [classes[i] for i in sample[1]['labels']], width=4
    ).permute(1, 2, 0))
    plt.show()

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    from utils import collate_fn
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
    
    device = torch.device("cuda") 
    def train_one_epoch(model, optimizer, loader, device, epoch):
        model.to(device)
        model.train()
        
        all_losses = [0]
        all_losses_dict = [0]
        
        for images, targets in tqdm(loader):
            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets

            losses = sum(loss for loss in loss_dict.values())
            loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
            loss_value = losses.item()
            
            all_losses[0]=loss_value
            all_losses_dict[0] = loss_dict_append

            # if not math.isfinite(loss_value):
            #     print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            #     print(loss_dict)
            #     sys.exit(1)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
        all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
        print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_classifier'].mean(),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean()
        ))

    num_epochs=10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

    torch.save(model.state_dict(), f'soy_res50.pt')
    
    model.eval()
    torch.cuda.empty_cache()
    test_dataset = CustomDataset(root=os.path.join(base_dir, "rcnn_aqua", dataset_path), split="test", transforms=get_transforms(False))

    img, _ = test_dataset[5]
    img_int = torch.tensor(img*255, dtype=torch.uint8)
    with torch.no_grad():
        prediction = model([img.to(device)])
        pred = prediction[0]

    fig = plt.figure(figsize=(14, 10))
    plt.imshow(draw_bounding_boxes(img_int,
        pred['boxes'][pred['scores'] > 0.8],
        [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], width=4
    ).permute(1, 2, 0))
    plt.show()

    def evaluate_model(model, test_dataset, device, threshold=0.5):
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

                # Count true positives, false positives, and false negatives
                for i in range(len(pred_boxes)):
                    max_iou = iou[i].max().item()
                    if max_iou >= 0.5 and pred_labels[i] == gt_labels[i]:
                        true_positives += 1
                    elif max_iou < 0.5 and pred_labels[i] == gt_labels[i]:
                        false_negatives += 1
                    elif max_iou >= 0.5 and pred_labels[i] != gt_labels[i]:
                        false_positives += 1

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)

        return precision, recall

    precision, recall = evaluate_model(model, test_dataset, device)
    print(prediction)
    print(recall)