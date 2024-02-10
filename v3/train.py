import config
import torch
from dataset import Dataset
from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.optim as optim 
from tqdm import tqdm
import config
from model import YOLOv3
from loss import YOLOLoss
from utils import (
	save_checkpoint,
)

# Creating a dataset object 
dataset = Dataset( 
	csv_file= config.train_csv_file,
	image_dir= config.image_dir,
	label_dir= config.label_dir,
	grid_sizes=[13, 26, 52], 
	anchors=config.ANCHORS, 
	transform=config.test_transform 
) 

# Creating a dataloader object 
loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=1, 
	shuffle=True, 
) 

# # Defining the grid size and the scaled anchors 
# GRID_SIZE = [13, 26, 52] 
# scaled_anchors = torch.tensor(config.ANCHORS) / ( 
# 	1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
# ) 

# # Getting a batch from the dataloader 
# x, y = next(iter(loader)) 

# # Getting the boxes coordinates from the labels 
# # and converting them into bounding boxes without scaling 
# boxes = [] 
# for i in range(y[0].shape[1]): 
# 	anchor = scaled_anchors[i] 
# 	boxes += config.convert_cells_to_bboxes( 
# 			y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor 
# 			)[0] 

# # Applying non-maximum suppression 
# boxes = config.nms(boxes, iou_threshold=1, threshold=0.7) 

# # Plotting the image with the bounding boxes 
# config.plot_image(x[0].permute(1,2,0).to("cpu"), boxes)


# Define the train function to train the model 
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors): 
	# Creating a progress bar 
	progress_bar = tqdm(loader, leave=True) 

	# Initializing a list to store the losses 
	losses = [] 

	# Iterating over the training data 
	for _, (x, y) in enumerate(progress_bar): 
		x = x.to(config.device) 
		y0, y1, y2 = ( 
			y[0].to(config.device), 
			y[1].to(config.device), 
			y[2].to(config.device), 
		) 

		with torch.cuda.amp.autocast(): 
			# Getting the model predictions 
			outputs = model(x) 
			# Calculating the loss at each scale 
			loss = ( 
				loss_fn(outputs[0], y0, scaled_anchors[0]) 
				+ loss_fn(outputs[1], y1, scaled_anchors[1]) 
				+ loss_fn(outputs[2], y2, scaled_anchors[2]) 
			) 

		# Add the loss to the list 
		losses.append(loss.item()) 

		# Reset gradients 
		optimizer.zero_grad() 

		# Backpropagate the loss 
		scaler.scale(loss).backward() 

		# Optimization step 
		scaler.step(optimizer) 

		# Update the scaler for next iteration 
		scaler.update() 

		# update progress bar with loss 
		mean_loss = sum(losses) / len(losses) 
		progress_bar.set_postfix(loss=mean_loss)

# Creating the model from YOLOv3 class 
model = YOLOv3().to(config.device) 

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = config.leanring_rate) 

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.cuda.amp.GradScaler() 

# Defining the train dataset 
train_dataset = Dataset( 
	csv_file="./data/pascal voc/train.csv", 
	image_dir="./data/pascal voc/images/", 
	label_dir="./data/pascal voc/labels/", 
	anchors=config.ANCHORS, 
	transform=config.train_transform 
) 

# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
	train_dataset, 
	batch_size = config.batch_size, 
	num_workers = 2, 
	shuffle = True, 
	pin_memory = True, 
) 

# Scaling the anchors 
scaled_anchors = ( 
	torch.tensor(config.ANCHORS) *
	torch.tensor(config.s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
).to(config.device) 

# Training the model 
for e in range(1, config.epochs+1): 
	print("Epoch:", e) 
	training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors) 

	# Saving the model 
	if config.save_model: 
		save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
