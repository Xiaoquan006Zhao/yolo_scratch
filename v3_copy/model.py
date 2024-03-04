import torch 
import torch.nn as nn 
  
import numpy as np 
import pandas as pd 
import config
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from Blocks.BasicBlock import ConvBNMish
from Blocks.CSP import CSPBlock

from Blocks.PAN import PAN
from Blocks.SPPF import SPPFBlock



class CNNBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
		super().__init__() 
		self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 
		self.use_batch_norm = use_batch_norm 

	def forward(self, x): 
		x = self.conv(x) 
		if self.use_batch_norm: 
			x = self.bn(x) 
			return self.activation(x) 
		else: 
			return x

class ResidualBlock(nn.Module): 
	def __init__(self, channels, use_residual=True, num_repeats=1): 
		super().__init__() 
		
		res_layers = [] 
		for _ in range(num_repeats): 
			res_layers += [ 
				nn.Sequential( 
					ConvBNMish(channels, out_channels=channels, kernel_size=1, stride=1, padding=0),
        			ConvBNMish(channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
					# nn.Conv2d(channels, channels // 2, kernel_size=1), 
					# nn.BatchNorm2d(channels // 2), 
					# nn.LeakyReLU(0.1), 
					# nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
					# nn.BatchNorm2d(channels), 
					# nn.LeakyReLU(0.1) 
				) 
			] 
		self.layers = nn.ModuleList(res_layers) 
		self.use_residual = use_residual 
		self.num_repeats = num_repeats 
	
	def forward(self, x): 
		for layer in self.layers:
			if self.use_residual:
				x = x + layer(x)
			else:
				x = layer(x)

		return x
	
class ScalePrediction(nn.Module): 
	def __init__(self, in_channels, num_classes): 
		super().__init__() 
		self.pred = nn.Sequential( 
			nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
			nn.BatchNorm2d(2*in_channels), 
			nn.LeakyReLU(0.1), 
			nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1), 
		) 
		self.num_classes = num_classes 
	
	def forward(self, x): 
		output = self.pred(x) 
		output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
		output = output.permute(0, 1, 3, 4, 2) 
		return output

class YOLOv3(nn.Module): 
	def __init__(self, in_channels=3, num_classes=20): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

		self.layers = nn.ModuleList([ 
			ConvBNMish(in_channels, 32, kernel_size=3, stride=1, padding=1), 
			ConvBNMish(32, 64, kernel_size=3, stride=2, padding=1), 
			CSPBlock(64, 64, bottleNeck_use_residual=True, BottleNeck_repeats=1),

			ConvBNMish(64, 128, kernel_size=3, stride=2, padding=1), 
			CSPBlock(128, 128, bottleNeck_use_residual=True, BottleNeck_repeats=2),

			ConvBNMish(128, 256, kernel_size=3, stride=2, padding=1), 
			CSPBlock(256, 256, bottleNeck_use_residual=True, BottleNeck_repeats=8),

			ConvBNMish(256, 512, kernel_size=3, stride=2, padding=1), 
			CSPBlock(512, 512, bottleNeck_use_residual=True, BottleNeck_repeats=8),

			ConvBNMish(512, 1024, kernel_size=3, stride=2, padding=1), 
			CSPBlock(1024, 1024, bottleNeck_use_residual=True, BottleNeck_repeats=4),

			SPPFBlock(1024, pool_size=5, pool_repeats=3),
			PAN(config.PAN_channels, num_classes=config.num_classes),
		]) 
	
	def forward(self, x): 
		route_connections = [] 

		for layer in self.layers: 
			if isinstance(layer, PAN):
				return layer(route_connections)
			
			x = layer(x) 
			
			if isinstance(layer, CSPBlock) and (layer.BottleNeck_repeats == 8): 
				route_connections.append(x) 

			if isinstance(layer, SPPFBlock): 
				route_connections.append(x) 
				
		# return outputs


