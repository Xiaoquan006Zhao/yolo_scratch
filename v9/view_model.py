from IPython.display import Image
import torchvision
from torchview import draw_graph
import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel

model_name = "yolov9-e.yaml"
yaml_file = f"cfg/models/v9/{model_name}"

test_shape = (1,3,640,640)
x = torch.randn(test_shape)

model = DetectionModel(cfg = yaml_file)
architecture = model_name
model_graph = draw_graph(model, input_size=(x.shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
model_graph.visual_graph