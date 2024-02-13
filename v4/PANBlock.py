import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicBlock import (
    CBMBlock,
)
from torchview import draw_graph
from DenseBlock import *

class DownSample(nn.Module):
    def __init__(self, target_size):
        super(DownSample, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        """
        1x1 conv layer to maintain consistent channel depth across FPN
        """
        return F.adaptive_avg_pool2d(x, output_size = self.target_size)

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
		output = self.pred(x) 
		output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
		output = output.permute(0, 1, 3, 4, 2) 
		return output

class PAN(nn.Module):
    def __init__(self, channels_list, num_classes=20):
        """
        channels_list: List of channel sizes for each feature map level.
        """
        super(PAN, self).__init__()
        self.num_classes = num_classes 

        self.layers = nn.ModuleList([ 
            CBMBlock(channels_list[2], channels_list[0], kernel_size=1), 
            nn.Upsample(scale_factor=2), 
            CBMBlock(channels_list[0]+channels_list[1], channels_list[0], kernel_size=1, route=True),
            CBMBlock(channels_list[0], channels_list[0], kernel_size=1),
            nn.Upsample(scale_factor=2),
            CBMBlock(channels_list[0]+channels_list[0], channels_list[0], kernel_size=1),
            ScalePrediction(channels_list[0], num_classes= self.num_classes),
            CBMBlock(channels_list[0], channels_list[0], kernel_size=1),
            nn.Upsample(scale_factor=0.5),
            CBMBlock(channels_list[0]+channels_list[0], channels_list[0], kernel_size=1),
            ScalePrediction(channels_list[0], num_classes= self.num_classes),
            CBMBlock(channels_list[0], channels_list[0], kernel_size=1),
            nn.Upsample(scale_factor=0.5),
            CBMBlock(channels_list[0]+channels_list[2], channels_list[0], kernel_size=1),
            ScalePrediction(channels_list[0], num_classes= self.num_classes),
        ])
    
    def forward(self, features):
        """
        features: List of feature maps from the backbone at different levels.
        """
        f1, f2, f3 = features  # High to low resolution
        x = f3
        outputs = [] 
        route_connections = [f1, f2] 

        for layer in self.layers: 
            if isinstance(layer, ScalePrediction): 
                outputs.append(layer(x)) 
                continue

            # CBMBlock will execute here as well
            # print(x.shape)
            x = layer(x) 

            if isinstance(layer, CBMBlock) and layer.route is True:
                route_connections.insert(0, x)

            elif isinstance(layer, nn.Upsample): 
                if len(route_connections) == 0:
                     route_connections.append(f3)
                x = torch.cat([x, route_connections[-1]], dim=1) 
                route_connections.pop() 

            # elif isinstance(layer, DownSample): 
            #     # last concate
            #     if len(route_connections) == 0:
            #          route_connections.append(f3)
            #     x = torch.cat([x, route_connections[-1]], dim=1) 
            #     route_connections.pop() 

        return outputs


def test_PAN():
    # Initialize the PAN model with a predefined list of channels
    pan = PAN([128, 256, 512], num_classes=20)
    
    # Create dummy feature maps as input
    f1 = torch.randn(1, 128, 52, 52)  # High resolution
    f2 = torch.randn(1, 256, 26, 26)  # Medium resolution
    f3 = torch.randn(1, 512, 13, 13)  # Low resolution
    features = [f1, f2, f3]

    # Pass the feature maps through the PAN model
    outputs = pan(features)
    
    # Print the shape of the outputs for verification
    for i, output in enumerate(outputs):
        print(f'Output {i+1} Shape: {output.shape}')

    return pan

# # Run the test function
# model = test_PAN()

# class PANWrapper(nn.Module):
#     def __init__(self, pan_model):
#         super().__init__()
#         self.pan_model = pan_model

#     def forward(self, *inputs):
#         # Package the multiple inputs into a list before forwarding them to the PAN model
#         features = list(inputs)
#         return self.pan_model(features)

# # Wrap your PAN model
# model = PANWrapper(model)

# architecture = 'denseblock'
# model_graph = draw_graph(model, 
#                          input_size=([(1, 128, 52, 52),(1, 256, 26, 26),(1, 512, 13, 13)]), 
#                          graph_dir ='TB' , 
#                          roll=True, 
#                          expand_nested=True, 
#                          graph_name=f'self_{architecture}',
#                          save_graph=True,filename=f'self_{architecture}')
# model_graph.visual_graph
