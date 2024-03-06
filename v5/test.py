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
    output = model(x) 

    # nms
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
        outputs = model(x) 
        # nms
        prediction_bboxes = [[] for _ in range(x.shape[0])] 
        target_bboxes = [[] for _ in range(x.shape[0])] 

        for i in range(3): 
            _, A, _, _, _ = output[i].shape 
            prediction_bboxes_scale_i = convert_cells_to_bboxes(output[i], config.scaled_anchors[i], config.grid_sizes[i]) 
            for index, (box) in enumerate(prediction_bboxes_scale_i): 
                prediction_bboxes[index] += box 
                target_bboxes[index] += y[i][index]

    for i in range(config.test_batch_size): 
        prediction_bboxes_nms_batch = nms(prediction_bboxes[i], config.enough_overlap_threshold, config.valid_prediction_threshold)
        target_bboxes_batch = [box for box in target_bboxes[i] if box[0] > config.valid_prediction_threshold]

        print(len(prediction_bboxes_nms_batch))
        print(len(target_bboxes_batch))

        print(len(prediction_bboxes_nms_batch[0]))
        print(len(target_bboxes_batch[0]))

        precision_batch, recall_batch = calculate_precision_recall(prediction_bboxes_nms_batch, target_bboxes_batch)
        precisions.append(precision_batch)
        recalls.append(recall_batch)
   
print(f"Precision:{sum(precisions)/len(precisions)}, Recall:{sum(recalls)/len(recalls)}")
model.train() 
