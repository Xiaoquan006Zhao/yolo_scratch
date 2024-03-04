import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 
  

which_dataset = "pascal voc"
class_labels = [ 
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

train_csv_file = f"../data/{which_dataset}/2examples.csv"
test_csv_file = f"../data/{which_dataset}/2examples.csv"
# train_csv_file = f"../data/{which_dataset}/train.csv"
# test_csv_file = f"../data/{which_dataset}/test.csv"
image_dir = f"../data/{which_dataset}/images/"
label_dir = f"../data/{which_dataset}/labels/"  

device = "cuda" if torch.cuda.is_available() else "cpu"
load_model = True
save_model = True
checkpoint_file = "checkpoint.pth.tar"
ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 

PAN_channels = [256, 512, 1024]

batch_size = 4
leanring_rate = 1e-4
epochs = 1000
image_size = 640
s = [image_size // 32, image_size // 16, image_size // 8] 

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



