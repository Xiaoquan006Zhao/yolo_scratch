import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 
import os
from kmeans_anchor import auto_anchor
from enum import Enum
import platform



# -------------------------------------- DATASET --------------------------------------
class_labels = [ 
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# class_labels = ["pod"]
num_classes = len(class_labels)

# -------------------------------------- DATA Location --------------------------------------
dataset = "pascal voc"
# dataset = "soybean"

system_type = platform.system()
if system_type == "Windows":
	base_dir = os.getcwd()
	#train_csv_file = os.path.join(base_dir, "data", dataset, "100examples.csv")
	#test_csv_file = os.path.join(base_dir, "data", dataset, "100examples_test.csv")
	train_csv_file = os.path.join(base_dir, "data", dataset, "train.csv")
	test_csv_file = os.path.join(base_dir, "data", dataset, "test.csv")
	
	train_image_dir = os.path.join(base_dir, "data", dataset, "train","images")
	train_label_dir = os.path.join(base_dir, "data", dataset, "train","labels")

	test_image_dir = os.path.join(base_dir, "data", dataset, "test","images")
	test_label_dir = os.path.join(base_dir, "data", dataset, "test","labels")

	checkpoint_file = os.path.join(base_dir, "v9", f"{dataset}_checkpoint.pth.tar")
else:
	train_csv_file = f"../data/{dataset}/8examples.csv"
	test_csv_file = f"../data/{dataset}/8examples_test.csv"
	# train_csv_file = f"../data/{dataset}/train.csv"
	# test_csv_file = f"../data/{dataset}/test.csv"

	train_image_dir = f"../data/{dataset}/images/"
	train_label_dir = f"../data/{dataset}/labels/"  

	test_image_dir = f"../data/{dataset}/images/"
	test_label_dir = f"../data/{dataset}/labels/"  

	checkpoint_file = f"{dataset}_checkpoint.pth.tar"

	if system_type == "Darwin":
		train_csv_file = f"../data/{dataset}/8examples.csv"
		test_csv_file = f"../data/{dataset}/8examples_test.csv"

# -------------------------------------- Model Parameter --------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 2 if device == "cuda" else 0

load_model = True
save_model = True
train_batch_size = 4
batch_accumulation_steps = 4
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
num_anchors = 3

ANCHORS = auto_anchor(num_anchors, train_label_dir, grid_sizes)

# ANCHORS = [ 
# 	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
# 	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
# 	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
# ] 

scaled_anchors = ( 
    torch.tensor(ANCHORS, dtype=torch.float32) *
    torch.tensor(grid_sizes, dtype=torch.float32).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(device)


PAN_channels = [256, 512, 1024]

# -------------------------------------- Augmentation --------------------------------------
train_transform = A.Compose( 
	[ 
		A.LongestMaxSize(max_size=image_size), 
		A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT), 
		A.ColorJitter( 
			brightness=0.5, contrast=0.5, 
			saturation=0.5, hue=0.5, p=0.5
		), 
		A.HorizontalFlip(p=0.5), 
		A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255), 
		ToTensorV2() 
	], 
	bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]) 
) 

test_transform = A.Compose( 
	[ 
		A.LongestMaxSize(max_size=image_size), 
		A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT), 
		A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255), 
		ToTensorV2() 
	], 
	bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]) 
)



