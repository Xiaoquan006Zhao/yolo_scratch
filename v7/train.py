from config import Config
import torch
from dataset import Dataset
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.optim as optim 
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import YOLOv7
from loss import YOLOLoss
from utils import (
	load_checkpoint,
	save_checkpoint,
)

# Define the train function to train the model 
def training_loop(e, loader, model, optimizer, scheduler, loss_fn, scaler, scaled_anchors, scales):
	iters = len(loader) 
	# Creating a progress bar 
	progress_bar = tqdm(loader, leave=True) 

	# Initializing a list to store the losses 
	losses = [] 

	# Iterating over the training data 
	for i, (x, y) in enumerate(progress_bar): 
		x = x.to(Config.device) 
		y0, y1, y2 = ( 
			y[0].to(Config.device), 
			y[1].to(Config.device), 
			y[2].to(Config.device), 
		) 

		with torch.cuda.amp.autocast(): 
			# Getting the model predictions 
			outputs = model(x) 

			# Calculating the loss at each scale 
			# the weight [4, 1, 0.4] is found in https://docs.ultralytics.com/yolov5/tutorials/architecture_description/#41-compute-losses
			loss = ( 
				loss_fn(outputs[0], y0, scaled_anchors[0], scales[0]) 
				+ loss_fn(outputs[1], y1, scaled_anchors[1], scales[1]) 
				+ loss_fn(outputs[2], y2, scaled_anchors[2], scales[2]) 
			) 

		# Add the loss to the list 
		losses.append(loss.item()) 

		# Reset gradients 
		optimizer.zero_grad() 

		# Backpropagate the loss 
		scaler.scale(loss).backward() 

		# Optimization step 
		scaler.step(optimizer) 

		scheduler.step(e + i / iters)

		# Update the scaler for next iteration 
		scaler.update() 

		# update progress bar with loss 
		mean_loss = sum(losses) / len(losses) 
		progress_bar.set_postfix(loss=mean_loss)

# Creating the model from YOLOv3 class 
model = YOLOv5().to(Config.device) 

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = Config.max_leanring_rate) 

scheduler = CosineAnnealingWarmRestarts(optimizer, 
										T_0 = 32,
										T_mult = 1, # A factor increases TiTi​ after a restart
										eta_min = Config.min_leanring_rate) 
scheduler.base_lrs[0] = Config.max_leanring_rate

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.cuda.amp.GradScaler() 

if Config.load_model: 
	load_checkpoint(Config.checkpoint_file, model, optimizer, Config.max_leanring_rate) 

# Defining the train dataset 
train_dataset = Dataset( 
	csv_file = Config.train_csv_file,
	image_dir = Config.image_dir,
	label_dir = Config.label_dir,
	anchors=Config.ANCHORS, 
	image_size = Config.image_size, 
	grid_sizes = Config.s, 
	transform=Config.train_transform 
) 

# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
	train_dataset, 
	batch_size = Config.batch_size, 
	num_workers = 2, 
	shuffle = True, 
	pin_memory = True, 
) 
# Training the model 
for e in range(1, Config.epochs+1): 
	print("Epoch:", e) 
	training_loop(e, train_loader, model, optimizer, scheduler, loss_fn, scaler, Config.scaled_anchors, Config.s) 

	# Saving the model 
	if Config.save_model: 
		save_checkpoint(model, optimizer, filename=Config.checkpoint_file)
