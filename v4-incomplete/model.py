import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import config

from BasicBlock import CBMBlock
from CSPBlock import CSPBlock
from PANBlock import PAN
from SPPBlock import SPPBlock
from DenseBlock import DenseBlock

# Defining CNN Block 
class CNNBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
		super().__init__() 
		self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 
		self.use_batch_norm = use_batch_norm 

	def forward(self, x): 
		# Applying convolution 
		x = self.conv(x) 
		# Applying BatchNorm and activation if needed 
		if self.use_batch_norm: 
			x = self.bn(x) 
			return self.activation(x) 
		else: 
			return x

# Defining residual block 
class ResidualBlock(nn.Module): 
	def __init__(self, channels, use_residual=True, num_repeats=1): 
		super().__init__() 
		
		# Defining all the layers in a list and adding them based on number of 
		# repeats mentioned in the design 
		res_layers = [] 
		for _ in range(num_repeats): 
			res_layers += [ 
				nn.Sequential( 
					nn.Conv2d(channels, channels // 2, kernel_size=1), 
					nn.BatchNorm2d(channels // 2), 
					nn.LeakyReLU(0.1), 
					nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
					nn.BatchNorm2d(channels), 
					nn.LeakyReLU(0.1) 
				) 
			] 
		self.layers = nn.ModuleList(res_layers) 
		self.use_residual = use_residual 
		self.num_repeats = num_repeats 
	
	# Defining forward pass 
	def forward(self, x): 
		for layer in self.layers:
			if self.use_residual:
				x = x + layer(x)
			else:
				x = layer(x)

		return x
	
# Defining scale prediction class 
class ScalePrediction(nn.Module): 
	def __init__(self, in_channels, num_classes): 
		super().__init__() 
		# Defining the layers in the network 
		self.pred = nn.Sequential( 
			nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
			nn.BatchNorm2d(2*in_channels), 
			nn.LeakyReLU(0.1), 
			nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1), 
		) 
		self.num_classes = num_classes 
	
	# Defining the forward pass and reshaping the output to the desired output 
	# format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
	def forward(self, x): 
		# output = self.pred(x) 
		# output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
		# output = output.permute(0, 1, 3, 4, 2) 

		# just return x because we are v4 not v3, model prediction does not happen here
		return x

# Class for defining YOLOv3 model 
class YOLOv4(nn.Module): 
	def __init__(self, in_channels=3, num_classes=config.num_classes): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

		self.dense_block_1 = DenseBlock(layer_num=1, in_channels=64//2)
		self.dense_block_2 = DenseBlock(layer_num=2, in_channels=128//2)
		self.dense_block_3 = DenseBlock(layer_num=8, in_channels=256//2)
		self.dense_block_4 = DenseBlock(layer_num=8, in_channels=512//2)
		self.dense_block_5 = DenseBlock(layer_num=4, in_channels=1024//2)

		# Layers list for YOLOv3 
		self.layers = nn.ModuleList([ 
			CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1), 
			CNNBlock(32, 64, kernel_size=3, stride=2, padding=1), 
			CSPBlock(process_block=self.dense_block_1),
			CNNBlock(64, 128, kernel_size=3, stride=2, padding=1), 
			CSPBlock(process_block=self.dense_block_2),
			CNNBlock(128, 256, kernel_size=3, stride=2, padding=1), 
			CSPBlock(process_block=self.dense_block_3),
			ScalePrediction(256, num_classes=num_classes), 

			CNNBlock(256, 512, kernel_size=3, stride=2, padding=1), 
			CSPBlock(process_block=self.dense_block_4),
			ScalePrediction(512, num_classes=num_classes), 

			CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1), 
			CSPBlock(process_block=self.dense_block_5),
			ScalePrediction(1024, num_classes=num_classes), 

			SPPBlock(config.SPP_pool_sizes),
			CBMBlock(len(config.SPP_pool_sizes)*1024, 1024, 1),
			PAN(config.PAN_channels, num_classes=config.num_classes),
		]) 
	
	# Forward pass for YOLOv3 with route connections and scale predictions 
	def forward(self, x): 
		outputs = [] 
		route_connections = [] 

		for layer in self.layers: 
			if isinstance(layer, PAN):
				return layer(outputs)

			if isinstance(layer, ScalePrediction): 
				outputs.append(layer(x)) 
				continue
			
			x = layer(x) 

			if isinstance(layer, CSPBlock) and (layer.in_channels == 512 or layer.in_channels == 256): 
				route_connections.append(x) 
			
			elif isinstance(layer, nn.Upsample): 
				x = torch.cat([x, route_connections[-1]], dim=1) 
				route_connections.pop() 

		# Because return is done in PAN layer
		# return outputs



