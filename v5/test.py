import torch
from config import Config
from model import YOLOv5
from loss import YOLOLoss
import numpy as np
import torch.optim as optim 
from tqdm import tqdm
from utils import (
	load_checkpoint,	
	decodePrediction,
	plot_image,
	stable_divide,
	convert_cells_to_bboxes,
	nms,
)
from dataset import Dataset
from utils_metric import calculate_precision_recall

if __name__ == "__main__":
	model = YOLOv5().to(Config.device) 
	loss_fn = YOLOLoss() 
	optimizer = optim.Adam(model.parameters(), lr = Config.max_leanring_rate) 

	# optimizer = optim.Adam(
	# 	[
	# 		{'params': model.parameters()},
	# 		{'params': loss_fn.parameters(), 'lr': Config.min_leanring_rate}, 
	# 	],
	# 	lr=Config.max_leanring_rate
	# )

	scaler = torch.cuda.amp.GradScaler() 

	if Config.load_model: 
		load_checkpoint(Config.checkpoint_file, model, optimizer, Config.max_leanring_rate) 

	test_dataset = Dataset( 
		csv_file=Config.test_csv_file,
		image_dir=Config.image_dir,
		label_dir=Config.label_dir,
		anchors=Config.ANCHORS, 
		image_size = Config.image_size, 
		grid_sizes = Config.s, 
		num_classes= Config.num_classes,
		transform=Config.test_transform 
	) 

	test_loader = torch.utils.data.DataLoader( 
		test_dataset, 
		batch_size = Config.test_batch_size, 
		num_workers = 2, 
		shuffle = True, 
	) 

	x, y = next(iter(test_loader)) 
	x = x.to(Config.device) 

	model.eval() 
	with torch.no_grad(): 
		output = model(x) 

		bboxes = [[] for _ in range(x.shape[0])] 
		anchors = ( 
				torch.tensor(Config.ANCHORS) * torch.tensor(Config.s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
				).to(Config.device) 

		for i in range(3): 
			batch_size, A, S, _, _ = output[i].shape 
			anchor = anchors[i] 
			boxes_scale_i = convert_cells_to_bboxes(output[i], anchor, s=S, is_predictions=True) 
			for idx, (box) in enumerate(boxes_scale_i): 
				bboxes[idx] += box 
	model.train() 

	for i in range(batch_size): 
		nms_boxes = nms(bboxes[i], enough_overlap_threshold=0.5, valid_prediction_threshold=0.6) 

		plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

	true_positives = [[] for _ in range(Config.num_scales)]
	num_predictions = [[] for _ in range(Config.num_scales)]
	num_targets = [[] for _ in range(Config.num_scales)]

	progress_bar = tqdm(test_loader, leave=True) 
	for _, (x, y) in enumerate(progress_bar): 
		x = x.to(Config.device) 
		outputs = model(x) 

		for i in range(Config.num_scales):
			predictions = outputs[i]
			targets = y[i].to(Config.device)
			true_positives_batch, num_predictions_batch, num_targets_batch = calculate_precision_recall(predictions, targets, Config.scaled_anchors[i], Config.s[i])

			true_positives[i].append(true_positives_batch)
			num_predictions[i].append(num_predictions_batch)
			num_targets[i].append(num_targets_batch)
	model.train() 

	for i in range(Config.num_scales):
		precision = stable_divide(sum(true_positives[i]), sum(num_predictions[i]))
		recall = stable_divide(sum(true_positives[i]), sum(num_targets[i]))
		print(f"Precision:{precision}, Recall:{recall}")	
