import torch 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import config
import os


def iou(box1, box2, is_pred=True): 
	if is_pred: 
		b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
		b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
		b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
		b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

		b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
		b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
		b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
		b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

		x1 = torch.max(b1_x1, b2_x1) 
		y1 = torch.max(b1_y1, b2_y1) 
		x2 = torch.min(b1_x2, b2_x2) 
		y2 = torch.min(b1_y2, b2_y2) 
		
		intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 

		box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
		box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
		union = box1_area + box2_area - intersection 

		epsilon = 1e-6
		iou_score = intersection / (union + epsilon) 

		return iou_score 
	
	else: 
		intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1]) 
		box1_area = box1[..., 0] * box1[..., 1] 
		box2_area = box2[..., 0] * box2[..., 1] 
		union_area = box1_area + box2_area - intersection_area 

		iou_score = intersection_area / union_area 

		return iou_score

def nms(bboxes, enough_overlap_threshold, valid_prediction_threshold):
    bboxes = [box for box in bboxes if box[1] > valid_prediction_threshold]

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_nms = []
    while bboxes:
        first_box = bboxes.pop(0)
        bboxes_nms.append(first_box)

        # Keep only bounding boxes that do not overlap significantly with the first_box  
		# And skip for different classes, because prediction for different classes should be independent
		# Check convert_cells_to_bboxes method for why class_prediction is stored at index 0 and why bbox parameter is stored at index [2:]
        bboxes = [box for box in bboxes if box[0] != first_box[0] or iou(torch.tensor(first_box[2:]), torch.tensor(box[2:])) < enough_overlap_threshold]

    return bboxes_nms

def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True): 
	batch_size = predictions.shape[0] 
	num_anchors = len(anchors) 
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

	x = 1 / s * (box_predictions[..., 0:1] + cell_indices) 
	y = 1 / s * (box_predictions[..., 1:2] +
				cell_indices.permute(0, 1, 3, 2, 4)) 
	width_height = 1 / s * box_predictions[..., 2:4] 

	converted_bboxes = torch.cat((best_class, objectness, x, y, width_height), dim=-1).reshape(
		batch_size, num_anchors * s * s, 6) 

	return converted_bboxes.tolist()

def plot_image(image, boxes): 
	colour_map = plt.get_cmap("tab20b") 
	colors = [colour_map(i) for i in np.linspace(0, 1, len(config.class_labels))] 

	img = np.array(image) 
	h, w, _ = img.shape 

	fig, ax = plt.subplots(1) 

	ax.imshow(img) 

	for box in boxes: 
		class_pred = box[0] 
		box = box[2:] 
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2

		rect = patches.Rectangle( 
			(upper_left_x * w, upper_left_y * h), 
			box[2] * w, 
			box[3] * h, 
			linewidth=2, 
			edgecolor=colors[int(class_pred)], 
			facecolor="none", 
		) 
		
		ax.add_patch(rect) 
		plt.text( 
			upper_left_x * w, 
			upper_left_y * h, 
			s=config.class_labels[int(class_pred)], 
			color="white", 
			verticalalignment="top", 
			bbox={"color": colors[int(class_pred)], "pad": 0}, 
		) 

	plt.show()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): 
	print("==> Saving checkpoint") 
	checkpoint = { 
		"state_dict": model.state_dict(), 
		"optimizer": optimizer.state_dict(), 
	} 
	torch.save(checkpoint, filename)

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


