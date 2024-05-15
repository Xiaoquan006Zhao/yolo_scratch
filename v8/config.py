import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from kmeans_anchor import auto_anchor
from enum import Enum
import platform


# -------------------------------------- DATASET --------------------------------------
class_labels = []

# -------------------------------------- DATA Location --------------------------------------
# dataset = "pascal voc"
# dataset = "soybean"
dataset = "coco"

system_type = platform.system()
print(system_type)
if system_type in ["Windows", "indows"]:
    base_dir = os.getcwd()

    class_file = os.path.join(base_dir, "data", dataset, "class.txt")
    with open(class_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            class_labels.append(line.strip())
    num_classes = len(class_labels)
    print(f"{num_classes} classes")

    train = "train"
    test = "test"

    # train_csv_file = os.path.join(base_dir, "data", dataset, "100examples.csv")
    # test_csv_file = os.path.join(base_dir, "data", dataset, "100examples_test.csv")
    train_csv_file = os.path.join(base_dir, "data", dataset, f"{train}.csv")
    test_csv_file = os.path.join(base_dir, "data", dataset, f"{test}.csv")

    train_image_dir = os.path.join(base_dir, "data", dataset, f"{train}", "images")
    train_label_dir = os.path.join(base_dir, "data", dataset, f"{train}", "labels")

    test_image_dir = os.path.join(base_dir, "data", dataset, f"{test}", "images")
    test_label_dir = os.path.join(base_dir, "data", dataset, f"{test}", "labels")

    checkpoint_file = os.path.join(base_dir, "v8", f"{dataset}_checkpoint.pth.tar")
else:
    # train_csv_file = f"../data/{dataset}/8examples.csv"
    # test_csv_file = f"../data/{dataset}/8examples_test.csv"

    train_csv_file = f"../data/{dataset}/train.csv"
    test_csv_file = f"../data/{dataset}/test.csv"

    train_image_dir = f"../data/{dataset}/images/"
    train_label_dir = f"../data/{dataset}/labels/"

    test_image_dir = f"../data/{dataset}/images/"
    test_label_dir = f"../data/{dataset}/labels/"

    checkpoint_file = f"{dataset}_checkpoint.pth.tar"

# -------------------------------------- Model Parameter --------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 2 if device == "cuda" else 0

load_model = True
save_model = True
train_batch_size = 32
test_batch_size = 2
epochs = 1000
learning_rate = 1e-4
# 0.33 represents 50% overlap. Think of two boxes, side by side of the same size
# One covers half of the other, then we are left with 50% uncovered, 50% overlap, 50 uncovered
# Thus, 50% overlap = 0.33 IoU
enough_overlap_threshold = 0.33
valid_prediction_threshold = 0.75
numerical_stability = 1e-6


class CIOU_MODE(Enum):
    CI0U = 1
    IoU = 2
    WidthHeight = 3


image_size = 640
grid_sizes = [image_size // 32, image_size // 16, image_size // 8]

PAN_channels = [256, 512, 1024]
# -------------------------------------- Augmentation --------------------------------------
train_transform = A.Compose(
    [
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size + 20,
            min_width=image_size + 20,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        # A.ColorJitter(
        # 	brightness=0.5, contrast=0.5,
        # 	saturation=0.5, hue=0.5, p=0.5
        # ),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

test_transform = A.Compose(
    [
        # A.LongestMaxSize(max_size=image_size),
        # A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
