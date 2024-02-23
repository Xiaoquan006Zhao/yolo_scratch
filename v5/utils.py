import torch 
import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import config


# Defining a function to calculate Intersection over Union (IoU) 
def ciou(box1, box2, is_pred=True): 
	if is_pred: 
		# Convert from center to corner format
		b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2, box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
		b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2, box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2
		
		# Intersection area
		inter_area = torch.clamp(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1), min=0) * torch.clamp(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1), min=0)
		
		# Union area
		union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) + ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) - inter_area
		
		# IoU
		iou = inter_area / (union_area + config.numerical_stability)

		# Center distance
		center_distance = (box1[..., 0] - box2[..., 0])**2 + (box1[..., 1] - box2[..., 1])**2
		
		# Enclosing box
		c_x1, c_y1, c_x2, c_y2 = torch.min(b1_x1, b2_x1), torch.min(b1_y1, b2_y1), torch.max(b1_x2, b2_x2), torch.max(b1_y2, b2_y2)
		c_diag = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2
		
		# Aspect ratio
		v = (4 / (np.pi ** 2)) * ((torch.atan(box1[..., 2] / box1[..., 3]) - torch.atan(box2[..., 2] / box2[..., 3])) ** 2)
		alpha = v / (1 - iou + v + 1e-6)
		
		ciou_score = iou - (center_distance / (c_diag + 1e-6)) - alpha * v

		return ciou_score
	else: 
		# IoU score based on width and height of bounding boxes 
		
		# Calculate intersection area 
		intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1]) 

		# Calculate union area 
		box1_area = box1[..., 0] * box1[..., 1] 
		box2_area = box2[..., 0] * box2[..., 1] 
		union_area = box1_area + box2_area - intersection_area 

		# Calculate IoU score 
		iou_score = intersection_area / union_area 

		# Return IoU score 
		return iou_score

# Non-maximum suppression function to remove overlapping bounding boxes 
def nms(bboxes, enough_overlap_threshold, valid_prediction_threshold):
    # Filter out bounding boxes with objectness below the valid_prediction_threshold
	# Check decodePrediction method for why objectness is stored at index 1
    bboxes = [box for box in bboxes if box[1] > valid_prediction_threshold]

    # Sort the bounding boxes by confidence in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

	# Initialize the list of bounding boxes after non-maximum suppression. 
    bboxes_nms = []
    while bboxes:
        # Get the bounding box with the highest confidence
        first_box = bboxes.pop(0)
        bboxes_nms.append(first_box)

        # Keep only bounding boxes that do not overlap significantly with the first_box  
		# And skip for different classes, because prediction for different classes should be independent
		# Check decodePrediction for why class_prediction is stored at index 0 and why bbox parameter is stored at index [2:]
        bboxes = [box for box in bboxes if box[0] != first_box[0] or ciou(torch.tensor(first_box[2:]), torch.tensor(box[2:])) < enough_overlap_threshold]

    return bboxes_nms

def decodePrediction_bbox(predictions, scaled_anchors, s):
	box_predictions = predictions[..., 1:5] 
	box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
	box_predictions[..., 2:4] = torch.exp(box_predictions[..., 2:4]) * scaled_anchors

	# Calculate cell indices 
	cell_indices = ( 
		torch.arange(s) 
		.repeat(predictions.shape[0], 3, s, 1) 
		.unsqueeze(-1) 
		.to(predictions.device) 
	) 

	x = (box_predictions[..., 0:1] + cell_indices / s) 
	y = (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4) / s) 
	width, height = box_predictions[..., 2:3],  box_predictions[..., 3:4]

	# Adjusting predictions for box coordinates
	box_preds = torch.cat([x, y, width, height], dim=-1)

	return box_preds

# Function to convert cells to bounding boxes 
def decodePrediction(predictions, scaled_anchors, s): 
	# Batch size used on predictions 
	batch_size = predictions.shape[0] 
	# Number of anchors 
	num_anchors = 3

	scaled_anchors = scaled_anchors.reshape(1, len(scaled_anchors), 1, 1, 2) 
	box_preds = decodePrediction_bbox(predictions, scaled_anchors, s)
		
	objectness = torch.sigmoid(predictions[..., 0:1]) 
	best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) 

	# Concatinating the values and reshaping them in (BATCH_SIZE, num_anchors * S * S, 6) shape 
	decoded_bboxes = torch.cat((best_class, objectness, box_preds), dim=-1).reshape(
		batch_size, num_anchors  * s * s, 6) 

	# Returning the reshaped and converted bounding box list 
	return decoded_bboxes.tolist()

# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes): 
	# Getting the color map from matplotlib 
	colour_map = plt.get_cmap("tab20b") 
	# Getting 20 different colors from the color map for 20 different classes 
	colors = [colour_map(i) for i in np.linspace(0, 1, len(config.class_labels))] 

	# Reading the image with OpenCV 
	img = np.array(image) 
	# Getting the height and width of the image 
	h, w, _ = img.shape 

	# Create figure and axes 
	fig, ax = plt.subplots(1) 

	# Add image to plot 
	ax.imshow(img) 

	# Plotting the bounding boxes and labels over the image 
	for box in boxes: 
		# Get the class from the box 
		class_pred = box[0] 
		# Get the center x and y coordinates 
		box = box[2:] 
		# Get the upper left corner coordinates 
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2

		# Create a Rectangle patch with the bounding box 
		rect = patches.Rectangle( 
			(upper_left_x * w, upper_left_y * h), 
			box[2] * w, 
			box[3] * h, 
			linewidth=2, 
			edgecolor=colors[int(class_pred)], 
			facecolor="none", 
		) 
		
		# Add the patch to the Axes 
		ax.add_patch(rect) 
		
		# Add class name to the patch 
		plt.text( 
			upper_left_x * w, 
			upper_left_y * h, 
			s=config.class_labels[int(class_pred)], 
			color="white", 
			verticalalignment="top", 
			bbox={"color": colors[int(class_pred)], "pad": 0}, 
		) 

	# Display the plot 
	plt.show()

# Function to save checkpoint 
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): 
	print("==> Saving checkpoint") 
	checkpoint = { 
		"state_dict": model.state_dict(), 
		"optimizer": optimizer.state_dict(), 
	} 
	torch.save(checkpoint, filename)

# Function to load checkpoint 
def load_checkpoint(checkpoint_file, model, optimizer, lr): 
    if not os.path.exists(checkpoint_file):
        # If the checkpoint file does not exist, print a message and return early.
        print(f"==> Checkpoint file {checkpoint_file} does not exist. Skipping load.")
        return
    
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda')) # Assuming you are using a CUDA device, adjust accordingly.
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def delete_all_files_in_augmentation_folder():
	augmentation_folder = config.augmentation_folder
	# Check if the directory exists to avoid errors
	if os.path.exists(augmentation_folder):
		# List all files and directories in the augmentation folder
		for filename in os.listdir(augmentation_folder):
			file_path = os.path.join(augmentation_folder, filename)
			try:
				# Check if it is a file and then remove it
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
				# Optionally, if you want to remove directories as well, uncomment the following lines:
				# elif os.path.isdir(file_path):
				#     shutil.rmtree(file_path)
			except Exception as e:
				print(f'Failed to delete {file_path}. Reason: {e}')
	else:
		print(f"The directory {augmentation_folder} does not exist.")



