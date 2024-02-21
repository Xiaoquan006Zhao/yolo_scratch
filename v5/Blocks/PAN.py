import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicBlock import (
    CBMBlock,
)
from torchview import draw_graph
from BasicBlock import ConvBNMish
from CSP import CSPBlock
from torchvision.ops import drop_block2d

class PAN(nn.Module):
    def __init__(self, channels_list, num_classes, use_dropblock=True, dropblock_params={'block_size': 5, 'p': 0.1}):
        """
        channels_list: List of channel sizes for each feature map level, later index represent deeper result
        """

        super(PAN, self).__init__()
        self.num_classes = num_classes 
        self.use_dropblock = use_dropblock
        self.dropblock_params = dropblock_params

        # 3 anchors, bbox + objectness = 5
        prediction_channels = (num_classes + 5) * 3

        self.conv1 = ConvBNMish(channels_list[2], channels_list[1], kernel_size=1, stride=1,padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2), 
        self.CSP1 = CSPBlock(channels_list[1], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.conv2 = ConvBNMish(channels_list[1], channels_list[0], kernel_size=1, stride=1,padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2), 
        self.CSP2 = CSPBlock(channels_list[0], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.predication1 = nn.Conv2d(channels_list[0], prediction_channels, kernel_size=1, stride=1, padding=0)

        self.downsample1 = ConvBNMish(channels_list[0], channels_list[0], kernel_size=3, stride=2, padding=1)
        self.CSP3 = CSPBlock(channels_list[1], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.predication2 = nn.Conv2d(channels_list[1], prediction_channels, kernel_size=1, stride=1, padding=0)

        self.downsample2 = ConvBNMish(channels_list[1], channels_list[1], kernel_size=3, stride=2, padding=1)
        self.CSP4 = CSPBlock(channels_list[2], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.predication3 = nn.Conv2d(channels_list[2], prediction_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, features):
        """
        features: List of feature maps from the backbone at different levels.
        """
        f1, f2, f3 = features  # less channels to more channels
        x = f3
        outputs = [] 
        route_connections = [f1, f2] 

        x = self.conv1(x)
        route_connections.insert(x, 0)
        x = self.upsample1(x)
        x = torch.cat((x, route_connections.pop()), dim=1)
        x = self.CSP1(x)

        x = self.conv2(x)
        route_connections.insert(x, 0)
        x = self.upsample2(x)
        x = torch.cat((x, route_connections.pop()), dim=1)
        x = self.CSP2(x)

        outputs.append(self.predication1(x))

        x = self.downsample1(x)
        x = torch.cat((x, route_connections.pop()), dim=1)
        x = self.CSP3(x)

        outputs.append(self.predication2(x))

        x = self.downsample2(x)
        x = torch.cat((x, route_connections.pop()), dim=1)
        x = self.CSP4(x)

        outputs.append(self.predication3(x))

        return outputs[::-1] # predictions returned from more channels to less channels (big to small)


# def test_PAN():
#     # Initialize the PAN model with a predefined list of channels
#     pan = PAN([128, 256, 512], num_classes=20)
    
#     # Create dummy feature maps as input
#     f1 = torch.randn(1, 128, 52, 52)  # High resolution
#     f2 = torch.randn(1, 256, 26, 26)  # Medium resolution
#     f3 = torch.randn(1, 512, 13, 13)  # Low resolution
#     features = [f1, f2, f3]

#     # Pass the feature maps through the PAN model
#     outputs = pan(features)
    
#     # Print the shape of the outputs for verification
#     for i, output in enumerate(outputs):
#         print(f'Output {i+1} Shape: {output.shape}')

#     return pan

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
