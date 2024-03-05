import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 
import os
  
print("Initializing config...")

dataset = "pascal voc"
if os.name == 'nt':
	base_dir = os.getcwd()
	#train_csv_file = os.path.join(base_dir, "data", dataset, "100examples.csv")
	#test_csv_file = os.path.join(base_dir, "data", dataset, "100examples_test.csv")
	train_csv_file = os.path.join(base_dir, "data", dataset, "train.csv")
	test_csv_file = os.path.join(base_dir, "data", dataset, "test.csv")
	image_dir = os.path.join(base_dir, "data", dataset, "images")
	label_dir = os.path.join(base_dir, "data", dataset, "labels")
	checkpoint_file = os.path.join(base_dir, "v5", f"{dataset}_checkpoint.pth.tar")
else:
	train_csv_file = f"../data/{dataset}/2examples.csv"
	test_csv_file = f"../data/{dataset}/2examples_test.csv"
	#train_csv_file = f"../data/{dataset}/train.csv"
	#test_csv_file = f"../data/{dataset}/test.csv"
	image_dir = f"../data/{dataset}/images/"
	label_dir = f"../data/{dataset}/labels/"  
	checkpoint_file = f"{dataset}_checkpoint.pth.tar"

PAN_channels = [256, 512, 1024]

load_model = True
save_model = True

epochs = 1000
batch_size = 4
test_batch_size = 2
min_leanring_rate = 1e-5 * 5
max_leanring_rate = min_leanring_rate
numerical_stability = 1e-6
image_size = 640
s = [image_size // 32, image_size // 16, image_size // 8] 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-{device}-")

num_anchors = 3
# ANCHORS = auto_anchor(num_anchors, label_dir, s)
num_scales = len(s)

ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 

scaled_anchors = ( 
	torch.tensor(ANCHORS) *
	torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
).to(device) 

class_labels = [ 
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
num_classes = len(class_labels)

valid_prediction_threshold = 0.8
enough_overlap_threshold = 0.6

train_transform = A.Compose( 
	[ 
		A.Resize(height=image_size, width=image_size),
		A.ColorJitter(
			brightness=0.5, contrast=0.5,
			saturation=0.5, hue=0.5, p=0.5
		),
		A.Rotate(limit=45, p=0.5),
		A.HorizontalFlip(p=0.5),
		A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  
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