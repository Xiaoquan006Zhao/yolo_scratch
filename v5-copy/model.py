import torch.nn as nn 
from config import Config

from Blocks.BasicBlock import ConvBNMish
from Blocks.CSP import CSPBlock
from Blocks.PAN import PAN
from Blocks.SPPF import SPPFBlock

class YOLOv5(nn.Module): 
	def __init__(self, in_channels=3, num_classes=Config.num_classes): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

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
			PAN(Config.PAN_channels, num_classes=Config.num_classes),
		]) 
	
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


