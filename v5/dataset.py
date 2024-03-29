import torch 
import pandas as pd 
import os 
import numpy as np
import config
from PIL import Image 
from utils import (
	ciou,
)

class Dataset(torch.utils.data.Dataset): 
	def __init__( 
		self, csv_file, image_dir, label_dir, anchors, 
		image_size, grid_sizes, 
		num_classes, transform=None
	): 
		self.label_list = pd.read_csv(csv_file) 
		self.image_dir = image_dir 
		self.label_dir = label_dir 
		self.image_size = image_size 
		self.transform = transform 
		self.grid_sizes = grid_sizes 
		self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) 
		self.num_anchors = self.anchors.shape[0] 
		self.num_anchors_per_scale = self.num_anchors // 3
		self.num_classes = num_classes 

		self.ignore_iou_thresh = 0.5

	def __len__(self): 
		return len(self.label_list) 
	
	def __getitem__(self, idx): 
		label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1]) 
		# We are applying roll to move class label to the last column 
		# 5 columns: x, y, width, height, class_label 
		bboxes = np.roll(np.loadtxt(fname=label_path, 
						delimiter=" ", ndmin=2), 4, axis=1).tolist() 
		
		img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0]) 
		image = np.array(Image.open(img_path).convert("RGB")) 

		if self.transform: 
			augs = self.transform(image=image, bboxes=bboxes) 
			image = augs["image"] 
			bboxes = augs["bboxes"] 

		targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
				for s in self.grid_sizes] 
		
		for box in bboxes: 
			# Calculate iou of bounding box with anchor boxes
			# Because it's not prediction, we only pass in width and height (aspect ratio)
			iou_anchors = ciou(torch.tensor(box[2:4]), self.anchors, config.CIOU_MODE.WidthHeight) 

			# Selecting the best anchor box and maintain the order from [0 to 8]
			# Since the config.Anchors is organized in 3x3 mannner, 
			# the iou_anchors calculation return the iou score from least scale to most scale [13, 26, 52]
			# so later when dividing by self.num_anchors_per_scale, we can get the scale of current anchor
			anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
			x, y, width, height, class_label = box 

			# At each scale, assigning the bounding box to the best matching anchor box 
			has_anchor = [False] * 3
			for anchor_idx in anchor_indices: 
				scale_idx = anchor_idx // self.num_anchors_per_scale 
				anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
				
				s = self.grid_sizes[scale_idx] 
				i, j = int(s * y), int(s * x) 
				anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 

				# In the following situations we do not consider if the anchor box within grid is already assigned
				# Because the anchors are ordered by IoU score, 
				# thus the best anchors has already been assigned to a cell
				# And multiple anchor (aspect ratio) do not share value, we don't need to worry about overwriting

				# If the anchor box within grid is not assigned 
				if not anchor_taken and not has_anchor[scale_idx]: 
					targets[scale_idx][anchor_on_scale, i, j, 0] = 1

					# Calculating the center of the bounding box relative to the cell 
					x_cell, y_cell = s * x - j, s * y - i 
					# Calculating the width and height of the bounding box relative to the cell 
					width_cell, height_cell = width * s, height * s

					box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) 

					targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 
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

		return image, tuple(targets)


