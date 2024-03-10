import config
import torch
from dataset import Dataset
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.optim as optim 
from tqdm import tqdm
import config
from model import YOLOv5
from loss import YOLOLoss
from utils import (
	load_checkpoint,
	save_checkpoint,
)

def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors): 
	progress_bar = tqdm(loader, leave=True) 
	losses = [] 

	for _, (x, y) in enumerate(progress_bar): 
		x = x.to(config.device) 
		y0, y1, y2 = ( 
			y[0].to(config.device), 
			y[1].to(config.device), 
			y[2].to(config.device), 
		) 

		with torch.cuda.amp.autocast(): 
			outputs = model(x) 
			loss = ( 
				loss_fn(outputs[0], y0, scaled_anchors[0]) 
				+ loss_fn(outputs[1], y1, scaled_anchors[1]) 
				+ loss_fn(outputs[2], y2, scaled_anchors[2]) 
			) 

		losses.append(loss.item()) 
		optimizer.zero_grad() 
		scaler.scale(loss).backward() 
		scaler.step(optimizer) 
		scaler.update() 

		mean_loss = sum(losses) / len(losses) 
		progress_bar.set_postfix(loss=mean_loss)

model = YOLOv5(num_classes=len(config.class_labels)).to(config.device) 
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate) 
loss_fn = YOLOLoss() 
scaler = torch.cuda.amp.GradScaler() 

if config.load_model: 
	load_checkpoint(config.checkpoint_file, model, optimizer, config.learning_rate) 

train_dataset = Dataset( 
	csv_file = config.train_csv_file,
	image_dir = config.image_dir,
	label_dir = config.label_dir,
	anchors = config.ANCHORS, 
	image_size = config.image_dir,
	grid_sizes = config.grid_sizes,
	num_classes = config.num_classes,
	transform = config.train_transform 
) 

train_loader = torch.utils.data.DataLoader( 
	train_dataset, 
	batch_size = config.train_batch_size, 
	num_workers = config.num_workers, 
	shuffle = True, 
	pin_memory = True, 
) 

for e in range(1, config.epochs+1): 
	print("Epoch:", e) 
	training_loop(train_loader, model, optimizer, loss_fn, scaler, config.scaled_anchors) 

	if config.save_model and e % 100 == 0: 
		save_checkpoint(model, optimizer, checkpoint_file=config.checkpoint_file)