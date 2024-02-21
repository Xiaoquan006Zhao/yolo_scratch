import torch
import config
from model import YOLOv5
from loss import YOLOLoss
import torch.optim as optim 
from utils import (
	load_checkpoint,	
	convert_cells_to_bboxes,
	plot_image,
	nms,
)
from dataset import Dataset

# Taking a sample image and testing the model 
# Setting the load_model to True 
load_model = True

# Defining the model, optimizer, loss function and scaler 
model = YOLOv5().to(config.device) 
optimizer = optim.Adam(model.parameters(), lr = config.leanring_rate) 
loss_fn = YOLOLoss() 
scaler = torch.cuda.amp.GradScaler() 

# Loading the checkpoint 
if load_model: 
	load_checkpoint(config.checkpoint_file, model, optimizer, config.leanring_rate) 

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
	batch_size = 1, 
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
	anchors = ( 
			torch.tensor(config.ANCHORS) * torch.tensor(config.s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
			).to(config.device) 

	# Getting bounding boxes for each scale 
	for i in range(3): 
		batch_size, A, S, _, _ = output[i].shape 
		anchor = anchors[i] 
		boxes_scale_i = convert_cells_to_bboxes(output[i], anchor, s=S, is_predictions=True) 
		for idx, (box) in enumerate(boxes_scale_i): 
			bboxes[idx] += box 
model.train() 

# Plotting the image with bounding boxes for each image in the batch 
for i in range(batch_size): 
	# Applying non-max suppression to remove overlapping bounding boxes 
	nms_boxes = nms(bboxes[i], enough_overlap_threshold=0.5, valid_prediction_threshold=0.6) 

	# Plotting the image with bounding boxes 
	plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
