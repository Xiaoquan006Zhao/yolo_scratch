import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2   
from kmeans_anchor import auto_anchor
import os

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
	train_csv_file = f"../data/{dataset}/100examples.csv"
	test_csv_file = f"../data/{dataset}/100examples_test.csv"
	#train_csv_file = f"../data/{dataset}/train.csv"
	#test_csv_file = f"../data/{dataset}/test.csv"
	image_dir = f"../data/{dataset}/images/"
	label_dir = f"../data/{dataset}/labels/"  
	checkpoint_file = f"{dataset}_checkpoint.pth.tar"

PAN_channels = [256, 512, 1024]

# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-{device}-")

load_model = True
save_model = True

epochs = 600
batch_size = 4
min_leanring_rate = 1e-4
max_leanring_rate = min_leanring_rate * 5
numerical_stability = 1e-6

image_size = 640
# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 
# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 

num_anchors = 3

if __name__ == '__main__':
	ANCHORS = auto_anchor(num_anchors, label_dir, s)

	# ANCHORS = [ 
	# 	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	# 	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	# 	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
	# ] 

	scaled_anchors = ( 
		torch.tensor(ANCHORS) *
		torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
	).to(device) 

	# Class labels 
	class_labels = [ 
		"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
		"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
		"pottedplant", "sheep", "sofa", "train", "tvmonitor"
	]
	num_classes = len(class_labels)

	valid_prediction_threshold = 0.6
	enough_overlap_threshold = 0.6

	# Transform for training 
	train_transform = A.Compose( 
		[ 
			# just resize without respecting the aspect ratio and padding
			A.Resize(height=image_size, width=image_size),
			# A.LongestMaxSize(max_size=image_size),
			# A.PadIfNeeded(
			#     min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
			# ),
			A.ColorJitter(
				brightness=0.5, contrast=0.5,
				saturation=0.5, hue=0.5, p=0.5
			),
			# A.RandomCrop(width=256, height=256, p=0.5),
			A.Rotate(limit=45, p=0.5),
			A.HorizontalFlip(p=0.5),

			# Additional transformations
			#A.VerticalFlip(p=0.5),  # Flip the image vertically
			# Adjust brightness and contrast
			A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  
			# Add Gaussian noise
			#A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  
			# Affine transformations: shift, scale, rotate
			#A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  
			# Elastic deformation
			#A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),  
			
			# Normalize the image 
			A.Normalize( 
				mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
			), 
			# Convert the image to PyTorch tensor 
			ToTensorV2() 
		], 
		# Augmentation for bounding boxes 
		bbox_params=A.BboxParams( 
						format="yolo", 
						min_visibility=0.4, 
						label_fields=[] 
					) 
	) 

	# Transform for testing 
	test_transform = A.Compose( 
		[ 
			# Rescale an image so that maximum side is equal to image_size 
			A.LongestMaxSize(max_size=image_size), 
			# Pad remaining areas with zeros 
			A.PadIfNeeded( 
				min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
			), 
			# Normalize the image 
			A.Normalize( 
				mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
			), 
			# Convert the image to PyTorch tensor 
			ToTensorV2() 
		], 
		# Augmentation for bounding boxes 
		bbox_params=A.BboxParams( 
						format="yolo", 
						min_visibility=0.4, 
						label_fields=[] 
					) 
	)




