import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph

from .BasicBlock import ConvBNMish

class BottleNeck(nn.Module):
    def __init__(self, in_channels):
        super(BottleNeck,self).__init__()

        self.conv1 = ConvBNMish(2*in_channels, 2*in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNMish(2*in_channels, 2*in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBNMish(2*in_channels, 2*in_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBNMish(2*in_channels, 2*in_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBNMish(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        output1 = self.conv2(self.conv1(x))
        output2 = self.conv4(self.conv3(output1))

        return self.conv5(torch.cat((output1, output2), dim=1))

