import torch
import config
from model import YOLOv5
from loss import YOLOLoss
import numpy as np
import torch.optim as optim 
from tqdm import tqdm
from utils import (
	load_checkpoint,	
	decodePrediction,
	plot_image,
	nms,
)
from dataset import Dataset
from utils_metric import calculate_precision_recall


if __name__ == '__main__':
	# Defining the model, optimizer, loss function and scaler 
	model = YOLOv5().to(config.device) 
	optimizer = optim.Adam(model.parameters(), lr = config.max_leanring_rate) 
	loss_fn = YOLOLoss() 
	scaler = torch.cuda.amp.GradScaler() 

	# Loading the checkpoint 
	if config.load_model: 
		load_checkpoint(config.checkpoint_file, model, optimizer, config.max_leanring_rate) 

	# Defining the test dataset and data loader 
	test_dataset = Dataset( 
		csv_file=config.test_csv_file,
		image_dir=config.image_dir,
		label_dir=config.label_dir,
		anchors=config.ANCHORS, 
		image_size = config.image_size, 
		grid_sizes = config.s, 
		transform=config.test_transform 
	) 

	test_loader = torch.utils.data.DataLoader( 
		test_dataset, 
		batch_size = 4, 
		num_workers = 2, 
		shuffle = True, 
	) 

	# Getting a sample image from the test data loader 
	x, y = next(iter(test_loader)) 
	x = x.to(config.device) 

	model.eval() 
	with torch.no_grad(): 
		# Getting the model predictions 
		output = model(x) 

		# Getting the bounding boxes from the predictions 
		bboxes = [[] for _ in range(x.shape[0])] 

		# Getting bounding boxes for each scale 
		for i in range(3): 
			batch_size, A, grid_size, _, _ = output[i].shape 
			boxes_scale_i = decodePrediction(output[i], config.scaled_anchors[i], grid_size=grid_size) 
			for idx, (box) in enumerate(boxes_scale_i): 
				bboxes[idx] += box 
	model.train() 

	# Plotting the image with bounding boxes for each image in the batch 
	for i in range(batch_size): 
		# Applying non-max suppression to remove overlapping bounding boxes 
		nms_boxes = nms(bboxes[i]) 

		# Plotting the image with bounding boxes 
		plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

	precisions = [[] for _ in range(config.num_anchors)]
	recalls = [[] for _ in range(config.num_anchors)]


	model.eval() 
	progress_bar = tqdm(test_loader, leave=True) 
	for _, (x, y) in enumerate(progress_bar): 
		x = x.to(config.device) 
		outputs = model(x) 

		for i in range(config.num_anchors):
			predictions = outputs[i]
			targets = y[i].to(config.device)
			precision_batch, recall_batch = calculate_precision_recall(predictions, targets, config.scaled_anchors[i])
    
			precisions[i].append(precision_batch)
			recalls[i].append(recall_batch)
	model.train() 

	# for each scales
	for i in range(config.num_anchors):
		print(f"Precision:{sum(precisions[i])/len(precisions[i])}, Recall:{sum(recalls[i])/len(recalls[i])}")	
