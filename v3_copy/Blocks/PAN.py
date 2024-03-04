import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph

from .BasicBlock import ConvBNMish
from .CSP import CSPBlock
from torchvision.ops import drop_block2d

class ScalePrediction(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super(ScalePrediction, self).__init__()
        self.num_classes = num_classes
        out_channels = (num_classes + 5) * 3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    # Defining the forward pass and reshaping the output to the desired output 
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
    def forward(self, x): 
        output = self.conv1(x) 
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output = output.permute(0, 1, 3, 4, 2) 
        return output

class PAN(nn.Module):
    def __init__(self, channels_list, num_classes, use_dropblock=True, dropblock_params={'block_size': 3, 'p': 0.1}):
        """
        channels_list: List of channel sizes for each feature map level, later index represent deeper result
        """

        super(PAN, self).__init__()
        self.num_classes = num_classes 
        self.use_dropblock = use_dropblock
        self.dropblock_params = dropblock_params

        self.conv1 = ConvBNMish(channels_list[2], channels_list[1], kernel_size=1, stride=1,padding=0)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.CSP1 = CSPBlock(channels_list[2], channels_list[1], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.conv2 = ConvBNMish(channels_list[1], channels_list[0], kernel_size=1, stride=1,padding=0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.CSP2 = CSPBlock(channels_list[1], channels_list[0], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.predication1 = ScalePrediction(channels_list[0], self.num_classes)

        self.downsample1 = ConvBNMish(channels_list[0], channels_list[0], kernel_size=3, stride=2, padding=1)
        self.CSP3 = CSPBlock(channels_list[1], channels_list[1], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.predication2 = ScalePrediction(channels_list[1], self.num_classes)

        self.downsample2 = ConvBNMish(channels_list[1], channels_list[1], kernel_size=3, stride=2, padding=1)
        self.CSP4 = CSPBlock(channels_list[2], channels_list[2], bottleNeck_use_residual=False, BottleNeck_repeats=3)

        self.predication3 = ScalePrediction(channels_list[2], self.num_classes)

    def forward(self, features):
        """
        features: List of feature maps from the backbone at different levels.
        """
        f1, f2, f3 = features  # less channels to more channels
        x = f3
        outputs = [] 
        route_connections = [f1, f2] 

        x = self.conv1(x)
        route_connections.insert(0, x)
        x = self.upsample1(x)
        x = torch.cat([x, route_connections[-1]], dim=1)
        route_connections.pop()
        x = self.CSP1(x)

        x = self.conv2(x)
        route_connections.insert(1, x)
        x = self.upsample2(x)
        x = torch.cat([x, route_connections[-1]], dim=1)
        route_connections.pop()
        x = self.CSP2(x)
        # x = drop_block2d(x, **self.dropblock_params)

        outputs.append(self.predication1(x))

        x = self.downsample1(x)
        x = torch.cat([x, route_connections[-1]], dim=1)
        route_connections.pop()
        x = self.CSP3(x)
        # x = drop_block2d(x, **self.dropblock_params)

        outputs.append(self.predication2(x))

        x = self.downsample2(x)
        x = torch.cat([x, route_connections[-1]], dim=1)
        route_connections.pop()
        x = self.CSP4(x)
        # x = drop_block2d(x, **self.dropblock_params)

        outputs.append(self.predication3(x))

        return outputs[::-1] # predictions returned from more channels to less channels (big to small)



