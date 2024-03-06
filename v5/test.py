import torch
import config
from model import YOLOv3
from tqdm import tqdm
from loss import YOLOLoss
import torch.optim as optim 
from utils import (
    load_checkpoint,	
    convert_cells_to_bboxes,
    plot_image,
    nms,
)
from dataset import Dataset
from utils_metric import (
    calculate_precision_recall,
)


model = YOLOv3(num_classes=len(config.class_labels)).to(config.device) 
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate) 
loss_fn = YOLOLoss() 
scaler = torch.cuda.amp.GradScaler() 

if config.load_model: 
    load_checkpoint(config.checkpoint_file, model, optimizer, config.learning_rate) 

test_dataset = Dataset( 
    csv_file=config.test_csv_file,
    image_dir=config.image_dir,
    label_dir=config.label_dir,
    anchors=config.ANCHORS, 
    image_size = config.image_dir,
    grid_sizes = config.grid_sizes,
    num_classes = config.num_classes,
    transform=config.test_transform 
) 
test_loader = torch.utils.data.DataLoader( 
    test_dataset, 
    batch_size = config.test_batch_size, 
    num_workers = 2, 
    shuffle = True, 
) 

x, y = next(iter(test_loader)) 
x = x.to(config.device) 

model.eval() 
with torch.no_grad(): 
    # output shape (num_scale, batch, num_anchor, grid_size, grid_size, num_class+5)
    output = model(x) 

    # x shape (batch, num_anchor, grid_size, grid_size, num_class+5)
    bboxes = [[] for _ in range(x.shape[0])] 
    for i in range(3): 
        _, A, _, _, _ = output[i].shape 
        boxes_scale_i = convert_cells_to_bboxes(output[i], config.scaled_anchors[i], config.grid_sizes[i]) 
        for idx, (box) in enumerate(boxes_scale_i): 
            bboxes[idx] += box 

for i in range(config.test_batch_size): 
    nms_boxes = nms(bboxes[i], config.enough_overlap_threshold, config.valid_prediction_threshold) 
    plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

# ---------------------------------------- Precision & Recall ----------------------------------------
precisions = []
recalls = []

progress_bar = tqdm(test_loader, leave=True) 
for _, (x, y) in enumerate(progress_bar): 
    x = x.to(config.device) 

    with torch.no_grad(): 
        output = model(x) 
        prediction_bboxes = [[] for _ in range(x.shape[0])] 
        target_bboxes = [[] for _ in range(x.shape[0])] 

        for i in range(3): 
            _, A, _, _, _ = output[i].shape 
            prediction_bboxes_scale_i = convert_cells_to_bboxes(output[i], config.scaled_anchors[i], config.grid_sizes[i], to_list=False) 
            target_bboxes_scale_i = convert_cells_to_bboxes(y[i], config.scaled_anchors[i], config.grid_sizes[i], to_list=False) 

            for index in len(config.test_batch_size):
                prediction_bboxes[index].append(prediction_bboxes_scale_i)
                target_bboxes[index].append(target_bboxes_scale_i)

    for i in range(config.test_batch_size): 
        precision_batch, recall_batch = calculate_precision_recall(prediction_bboxes[i], target_bboxes[i])
        precisions.append(precision_batch)
        recalls.append(recall_batch)
   
print(f"Precision:{sum(precisions)/len(precisions)}, Recall:{sum(recalls)/len(recalls)}")
model.train() 
