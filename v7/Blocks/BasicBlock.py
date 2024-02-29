import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvBNMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNMish, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2*in_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(2*in_channels, out_channels, kernel_size=1, stride=0, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.mish = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.mish(self.bn(self.conv2(self.conv1(x))))
