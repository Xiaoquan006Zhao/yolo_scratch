import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import config

from Blocks.BasicBlock import ConvBNMish
from Blocks.CSP import CSPBlock
from Blocks.PAN import PAN
from Blocks.SPPF import SPPFBlock

# yolo v5 architecture defination
# https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png
# https://docs.ultralytics.com/yolov5/tutorials/architecture_description/#1-model-structure

# Class for defining YOLOv3 model 
class YOLOv5(nn.Module): 
	def __init__(self, in_channels=3, num_classes=config.num_classes): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

		# Layers list for YOLOv3 
		self.layers = nn.ModuleList([ 
			ConvBNMish(in_channels, 64, kernel_size=6, stride=2, padding=2), 
			ConvBNMish(64, 128, kernel_size=3, stride=2, padding=1), 
			CSPBlock(128, 128, bottleNeck_use_residual=True, BottleNeck_repeats=3),

			ConvBNMish(128, 256, kernel_size=3, stride=2, padding=1), 
			CSPBlock(256, 256, bottleNeck_use_residual=True, BottleNeck_repeats=6),

			ConvBNMish(256, 512, kernel_size=3, stride=2, padding=1), 
			CSPBlock(512, 512, bottleNeck_use_residual=True, BottleNeck_repeats=9),

			ConvBNMish(512, 1024, kernel_size=3, stride=2, padding=1), 
			CSPBlock(1024, 1024, bottleNeck_use_residual=True, BottleNeck_repeats=3),

			SPPFBlock(1024, pool_size=5, pool_repeats=3),
			PAN(config.PAN_channels, num_classes=config.num_classes),
		]) 
	
	# Forward pass for YOLOv3 with route connections and scale predictions 
	def forward(self, x): 
		outputs = [] 

		for layer in self.layers: 
			if isinstance(layer, PAN):
				return layer(outputs)
			
			x = layer(x) 

			if isinstance(layer, CSPBlock) and (layer.BottleNeck_repeats == 6 or layer.BottleNeck_repeats == 9): 
				outputs.append(x) 

			elif isinstance(layer, SPPFBlock):
				outputs.append(x)
				

		# Because return is done in PAN layer
		# return outputs

def test_DenseLayer():
    x = torch.randn(1, 3, config.image_size, config.image_size)
    model = YOLOv5().to(config.device) 
    # print(model(x).shape)
    # print(model)
    # del model
    return model

model = test_DenseLayer()

from IPython.display import Image
import torchvision
from torchview import draw_graph

architecture = 'denselayer'
model_graph = draw_graph(model, input_size=(1, 3, config.image_size, config.image_size), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
model_graph.visual_graph

