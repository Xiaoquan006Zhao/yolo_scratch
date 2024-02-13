import torch 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import config


# Defining a function to calculate Intersection over Union (IoU) 
def ciou(box1, box2, is_pred=True): 
	if is_pred: 
		# Calculate the center points of box1 and box2
		box1_center_x = box1[..., 0:1]
		box1_center_y = box1[..., 1:2]
		box2_center_x = box2[..., 0:1]
		box2_center_y = box2[..., 1:2]

		# Calculate the width and height of box1 and box2
		box1_width = box1[..., 2:3]
		box1_height = box1[..., 3:4]
		box2_width = box2[..., 2:3]
		box2_height = box2[..., 3:4]

		# Calculate the top left and bottom right corners of box1 and box2
		box1_x1 = box1_center_x - box1_width / 2
		box1_y1 = box1_center_y - box1_height / 2
		box1_x2 = box1_center_x + box1_width / 2
		box1_y2 = box1_center_y + box1_height / 2

		box2_x1 = box2_center_x - box2_width / 2
		box2_y1 = box2_center_y - box2_height / 2
		box2_x2 = box2_center_x + box2_width / 2
		box2_y2 = box2_center_y + box2_height / 2

		# Calculate the area of box1 and box2
		box1_area = box1_width * box1_height
		box2_area = box2_width * box2_height

		# Calculate intersection area
		inter_width = torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)
		inter_height = torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)
		inter_area = torch.clamp(inter_width, min=0) * torch.clamp(inter_height, min=0)

		# Calculate union area
		union_area = box1_area + box2_area - inter_area

		# Calculate IoU
		iou = inter_area / (union_area + 1e-6)

		# Calculate the center distance
		center_distance = torch.square(box1_center_x - box2_center_x) + torch.square(box1_center_y - box2_center_y)

		# Calculate the diagonal length of the smallest enclosing box covering both boxes
		enclosing_x_max = torch.max(box1_x2, box2_x2)
		enclosing_x_min = torch.min(box1_x1, box2_x1)
		enclosing_y_max = torch.max(box1_y2, box2_y2)
		enclosing_y_min = torch.min(box1_y1, box2_y1)
		diagonal_length = torch.square(enclosing_x_max - enclosing_x_min) + torch.square(enclosing_y_max - enclosing_y_min)

		# Calculate the aspect ratio term
		v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(box2_width / box2_height) - torch.atan2(box1_width, box1_height), 2)
		alpha = v / (1 - iou + v + 1e-6)

		# Calculate CIoU loss
		ciou_loss = 1 - iou + (center_distance / (diagonal_length + 1e-6)) + alpha * v

		return ciou_loss
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
	# Check convert_cells_to_bboxes method for why objectness is stored at index 1
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
		# Check convert_cells_to_bboxes method for why class_prediction is stored at index 0 and why bbox parameter is stored at index [2:]
        bboxes = [box for box in bboxes if box[0] != first_box[0] or ciou(torch.tensor(first_box[2:]), torch.tensor(box[2:])) < enough_overlap_threshold]

    return bboxes_nms

# Function to convert cells to bounding boxes 
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True): 
	# Batch size used on predictions 
	batch_size = predictions.shape[0] 
	# Number of anchors 
	num_anchors = len(anchors) 
	# List of all the predictions 
	box_predictions = predictions[..., 1:5] 

	# If the input is predictions then we will pass the x and y coordinate 
	# through sigmoid function and width and height to exponent function and 
	# calculate the score and best class. 
	if is_predictions: 
		anchors = anchors.reshape(1, len(anchors), 1, 1, 2) 
		box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) 
		box_predictions[..., 2:] = torch.exp( 
			box_predictions[..., 2:]) * anchors 
		objectness = torch.sigmoid(predictions[..., 0:1]) 
		best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) 
	
	# Else we will just calculate scores and best class. 
	else: 
		objectness = predictions[..., 0:1] 
		best_class = predictions[..., 5:6] 

	# Calculate cell indices 
	cell_indices = ( 
		torch.arange(s) 
		.repeat(predictions.shape[0], 3, s, 1) 
		.unsqueeze(-1) 
		.to(predictions.device) 
	) 

	# Calculate x, y, width and height with proper scaling 
	x = 1 / s * (box_predictions[..., 0:1] + cell_indices) 
	y = 1 / s * (box_predictions[..., 1:2] +
				cell_indices.permute(0, 1, 3, 2, 4)) 
	width_height = 1 / s * box_predictions[..., 2:4] 

	# Concatinating the values and reshaping them in (BATCH_SIZE, num_anchors * S * S, 6) shape 
	converted_bboxes = torch.cat((best_class, objectness, x, y, width_height), dim=-1).reshape(
		batch_size, num_anchors * s * s, 6) 

	# Returning the reshaped and converted bounding box list 
	return converted_bboxes.tolist()

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
	print("==> Loading checkpoint") 
	checkpoint = torch.load(checkpoint_file, map_location=config.device) 
	model.load_state_dict(checkpoint["state_dict"]) 
	optimizer.load_state_dict(checkpoint["optimizer"]) 

	for param_group in optimizer.param_groups: 
		param_group["lr"] = lr 





