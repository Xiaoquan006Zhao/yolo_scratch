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
nms,
)
from dataset import Dataset
from utils_metric import calculate_precision_recall

if __name__ == "__main__":
	model = YOLOv5().to(Config.device) 
	optimizer = optim.Adam(model.parameters(), lr = Config.max_leanring_rate) 
	loss_fn = YOLOLoss() 
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
		transform=Config.test_transform 
	) 

	test_loader = torch.utils.data.DataLoader( 
		test_dataset, 
		batch_size = 1, 
		num_workers = 2, 
		shuffle = True, 
	) 

	x, y = next(iter(test_loader)) 
	x = x.to(Config.device) 

	model.eval() 
	with torch.no_grad(): 
		output = model(x) 
		decoded = [[] for _ in range(x.shape[0])] 

		for i in range(3): 
			batch_size, _, grid_size, _, _ = output[i].shape 

			decoded_scale_i = decodePrediction(output[i], Config.scaled_anchors[i], grid_size=grid_size) 

			for batch, (d) in enumerate(decoded_scale_i): 
				decoded[batch] += (d) 

	for i in range(batch_size): 
		nms_boxes = nms(decoded[i]) 
		plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

	precisions = [[] for _ in range(Config.num_anchors)]
	recalls = [[] for _ in range(Config.num_anchors)]
	progress_bar = tqdm(test_loader, leave=True) 
	for _, (x, y) in enumerate(progress_bar): 
		x = x.to(Config.device) 
		outputs = model(x) 

		for i in range(Config.num_anchors):
			predictions = outputs[i]
			targets = y[i].to(Config.device)
			precision_batch, recall_batch = calculate_precision_recall(predictions, targets, Config.scaled_anchors[i], Config.s[i])

			precisions[i].append(precision_batch)
			recalls[i].append(recall_batch)
	model.train() 

	for i in range(Config.num_anchors):
		print(f"Precision:{sum(precisions[i])/len(precisions[i])}, Recall:{sum(recalls[i])/len(recalls[i])}")	
