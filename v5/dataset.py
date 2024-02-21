import torch 
import pandas as pd 
import os 
import numpy as np
from PIL import Image, ImageFile, ImageDraw
import config
import random
from utils import (
	ciou,
)

# Create a dataset class to load the images and labels from the folder 
class Dataset(torch.utils.data.Dataset): 
	def __init__( 
		self, csv_file, image_dir, label_dir, anchors, 
		image_size=416, grid_sizes=[13, 26, 52], 
		num_classes=config.num_classes, transform=None
	): 
		# Read the csv file with image names and labels 
		self.label_list = pd.read_csv(csv_file) 
		# Image and label directories 
		self.image_dir = image_dir 
		self.label_dir = label_dir 
		# Image size 
		self.image_size = image_size 
		# Transformations 
		self.transform = transform 
		# Grid sizes for each scale 
		self.grid_sizes = grid_sizes 
		# Anchor boxes 
		self.anchors = torch.tensor( 
			anchors[0] + anchors[1] + anchors[2]) 
		# Number of anchor boxes 
		self.num_anchors = self.anchors.shape[0] 
		# Number of anchor boxes per scale 
		self.num_anchors_per_scale = self.num_anchors // 3
		# Number of classes 
		self.num_classes = num_classes 
		# Ignore IoU threshold: See logic (and comments) of where this threshold is used to understand
		self.ignore_iou_thresh = 0.5

	def __len__(self): 
		return len(self.label_list) 
	
	def __getitem__(self, idx): 
		image, bboxes = self.parse_to_image_and_bboxes(idx)

		targets = self.bbox_to_grid(bboxes)

		# Return the image and the target 
		return image, tuple(targets)

	def bbox_to_grid(self, bboxes):
		# Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
		# target : [object probabilities, x, y, width, height, class_label] 
		targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
				for s in self.grid_sizes] 
		
		# Identify anchor box and cell for each bounding box 
		for box in bboxes: 
			# Calculate iou of bounding box with anchor boxes
			# Because it's not prediction, we only pass in width and height (aspect ratio)
			iou_anchors = ciou(torch.tensor(box[2:4]), self.anchors, is_pred=False) 

			# Selecting the best anchor box and maintain the order from [0 to 8]
			# Since the config.Anchors is organized in 3x3 mannner, 
			# the iou_anchors calculation return the iou score from least scale to most scale [13, 26, 52]
			# so later when dividing by self.num_anchors_per_scale, we can get the scale of current anchor
			anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
			x, y, width, height, class_label = box 

			# At each scale, assigning the bounding box to the 
			# best matching anchor box 
			has_anchor = [False] * 3
			for anchor_idx in anchor_indices: 
				# which scale of [13, 26, 52], more detail refer to anchor_indices calculation
				scale_idx = anchor_idx // self.num_anchors_per_scale 
				# which anchor which in the scale
				anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
				
				# Identifying the grid size for the scale
				s = self.grid_sizes[scale_idx] 
				
				# Identifying the cell to which the bounding box belongs 
				i, j = int(s * y), int(s * x) 

				# locate the anchor within scale and grid
				anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 

				# In the following situations we do not consider if the anchor box within grid is already assigned
				# Because the anchors are ordered by IoU score, 
				# thus the best anchors has already been assigned to a cell
				# And multiple anchor (aspect ratio) do not share value, we don't need to worry about overwriting

				# If the anchor box within grid is not assigned 
				if not anchor_taken and not has_anchor[scale_idx]: 
					# Set the probability to 1 
					targets[scale_idx][anchor_on_scale, i, j, 0] = 1

					# Calculating the center of the bounding box relative to the cell 
					x_cell, y_cell = s * x - j, s * y - i 
					# Calculating the width and height of the bounding box 
					# relative to the cell 
					width_cell, height_cell = width * s, height * s
					# Idnetify the box coordinates 
					box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) 

					# Assigning the box coordinates to the target 
					targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 
					# Assigning the class label to the target 
					targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 

					# Set the anchor box as assigned for the scale 
					# The current scale we have already identified the object, 
					# We can skip this scale later
					has_anchor[scale_idx] = True

				# If the anchor box within grid is not assigned, and If the IoU is higher than ignore threshold, 
				# it implies that the anchor box significantly overlaps with the ground truth box 
				# but is not the best fit (since the best fit was assigned in the first if statement)
				elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
					# Set the probability to -1 to ignore the anchor box 
					# This anchor box overlaps with a ground truth object, 
					# but not enough to be considered the primary box for detection, 
					# yet it shouldn't be treated as a complete negative example (background) either.
					# Because later in loss calcuation, we calculate loss based on objectness == 0 or == 1
					# thus -1 is ignored
					targets[scale_idx][anchor_on_scale, i, j, 0] = -1

		return targets

	def parse_to_image_and_bboxes(self, idx):
		# Getting the label path 
		label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1]) 
		# We are applying roll to move class label to the last column 
		# 5 columns: x, y, width, height, class_label 
		bboxes = np.roll(np.loadtxt(fname=label_path, 
						delimiter=" ", ndmin=2), 4, axis=1).tolist() 
		
		# Getting the image path 
		img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0]) 
		image = np.array(Image.open(img_path).convert("RGB")) 

		# Albumentations augmentations 
		if self.transform: 
			augs = self.transform(image=image, bboxes=bboxes) 
			image = augs["image"] 
			bboxes = augs["bboxes"] 
		
		return image, bboxes

	def save_augmented_image_with_bboxes(self, augmentation_folder_path):
		random_index = random.randint(0, len(self.label_list) - 1)
		img, bboxes = self.parse_to_image_and_bboxes(random_index)

		# Check if img is a Tensor and convert it to a numpy array
		if isinstance(img, torch.Tensor):
			# Ensure tensor is on CPU and convert to numpy
			img = img.cpu().numpy()

		# Convert the numpy image to a PIL Image
		# If the image was a tensor, its format is likely CHW (channels, height, width)
		# PIL expects HWC format, so we need to transpose the axes
		if img.ndim == 3:  # This means the image has channels
			# Convert CHW to HWC
			img = np.transpose(img, (1, 2, 0))
		img = Image.fromarray((img * 255).astype(np.uint8))

		# Create a drawing context
		draw = ImageDraw.Draw(img)

		img_width, img_height = img.size
		
		# Iterate over the bounding boxes
		for bbox in bboxes:
			# Given bbox format is [center_x, center_y, width, height]
			center_x, center_y, bbox_width, bbox_height = bbox[:4]
			
			# Convert normalized midpoints and dimensions to absolute pixel values
			abs_center_x = center_x * img_width
			abs_center_y = center_y * img_height
			abs_bbox_width = bbox_width * img_width
			abs_bbox_height = bbox_height * img_height
			
			# Calculate the corners of the bounding box
			abs_x_min = abs_center_x - (abs_bbox_width / 2)
			abs_y_min = abs_center_y - (abs_bbox_height / 2)
			abs_x_max = abs_center_x + (abs_bbox_width / 2)
			abs_y_max = abs_center_y + (abs_bbox_height / 2)
			
			# Draw the rectangle
			draw.rectangle(((abs_x_min, abs_y_min), (abs_x_max, abs_y_max)), outline="red")
		
		file_path = augmentation_folder_path + self.label_list.iloc[random_index, 0]
		# Ensure the directory exists before saving
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		
		# Save the image
		img.save(file_path)




