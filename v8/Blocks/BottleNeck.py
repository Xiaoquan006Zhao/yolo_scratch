import torch.nn as nn
from .BasicBlock import ConvBNMish

class BottleNeck(nn.Module):
    def __init__(self, in_channels, use_residual = True):
        super(BottleNeck,self).__init__()
        self.conv1 = ConvBNMish(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNMish(in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.use_residual = use_residual

    def forward(self,x):
        output = self.conv2(self.conv1(x))

        return output if not self.use_residual else x+output

