from IPython.display import Image
import torchvision
from torchview import draw_graph
import torch
import torch.nn as nn
import config
from model import YOLOv9


model_name = "yolov9-kyle.yaml"

test_shape = (1,3,640,640)
x = torch.randn(test_shape)

model = YOLOv9(num_classes=len(config.class_labels))
architecture = model_name
model_graph = draw_graph(model, input_size=(x.shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
model_graph.visual_graph