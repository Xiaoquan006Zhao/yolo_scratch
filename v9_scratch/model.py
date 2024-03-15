import torch.nn as nn 
import config
from Blocks.BasicBlock import ConvBNMish
from Blocks.CSP import CSPBlock
from Blocks.PAN import PAN
from Blocks.SPPELAN import SPPFBlock
from Blocks.ScaledPredictions import ScaledPrediction

class YOLOv9(nn.Module): 
	def __init__(self, in_channels=3, num_classes=20, TRAINING=True): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

		self.layers = nn.ModuleList([ 
			ConvBNMish(in_channels, 64, kernel_size=3, stride=2, padding=1), 
			ConvBNMish(64, 128, kernel_size=3, stride=2, padding=1), 

			CSPBlock(128, 128, bottleNeck_use_residual=True, BottleNeck_repeats=3),

			ConvBNMish(128, 256, kernel_size=3, stride=2, padding=1), 
			CSPBlock(256, 256, bottleNeck_use_residual=True, BottleNeck_repeats=6),
			ScaledPrediction(256, self.num_classes),

			ConvBNMish(256, 512, kernel_size=3, stride=2, padding=1), 
			CSPBlock(512, 512, bottleNeck_use_residual=True, BottleNeck_repeats=9),
			ScaledPrediction(512, self.num_classes),

			ConvBNMish(512, 1024, kernel_size=3, stride=2, padding=1), 
			CSPBlock(1024, 1024, bottleNeck_use_residual=True, BottleNeck_repeats=4),
            ScaledPrediction(1024, self.num_classes),
			
			# ConvBNMish(in_channels, 64, kernel_size=3, stride=2, padding=1), 
			# ConvBNMish(64, 128, kernel_size=3, stride=2, padding=1), 

			# CSPBlock(128, 128, bottleNeck_use_residual=True, BottleNeck_repeats=3),
			# ConvBNMish(128, 256, kernel_size=3, stride=2, padding=1), 
			# CSPBlock(256, 256, bottleNeck_use_residual=True, BottleNeck_repeats=6),
			# ConvBNMish(256, 512, kernel_size=3, stride=2, padding=1), 
			# CSPBlock(512, 512, bottleNeck_use_residual=True, BottleNeck_repeats=9),
			# ConvBNMish(512, 1024, kernel_size=3, stride=2, padding=1), 
			# CSPBlock(1024, 1024, bottleNeck_use_residual=True, BottleNeck_repeats=4),

			# SPPFBlock(1024, pool_size=5, pool_repeats=3),
			# PAN(config.PAN_channels, num_classes=config.num_classes),
		]) 
	
	def forward(self, x): 
		outputs = [] 
		

		for layer in self.layers: 
			if isinstance(layer, ScaledPrediction):
				outputs.append(layer(x))
				continue
			
			x = layer(x) 
			
			# if isinstance(layer, CSPBlock) and (layer.BottleNeck_repeats == 6 or layer.BottleNeck_repeats == 9): 
			# 	route_connections.append(x) 

			# if isinstance(layer, SPPFBlock): 
			# 	route_connections.append(x) 
				
		return outputs[::-1]


