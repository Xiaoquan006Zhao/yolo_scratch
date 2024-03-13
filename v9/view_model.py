from IPython.display import Image
import torchvision
from torchview import draw_graph
import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel

yaml_file = "cfg/models/v9/yolov9e.yaml"

test_shape = (1,3,224,224)
x = torch.randn(test_shape)
model = DetectionModel(cfg = yaml_file)

architecture = 'denseblock'
model_graph = draw_graph(model, input_size=(x.shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
model_graph.visual_graph