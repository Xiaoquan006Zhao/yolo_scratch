import torch
import torch.nn as nn
from .BottleNeck import BottleNeck
from .BasicBlock import ConvBNMish

class EELANBlock(nn.Module):
    def __init__(self, in_channels, BottleNeck_repeats):
        super(EELANBlock, self).__init__()

        self.group_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.group_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.process_blocks = nn.Sequential(
            *[BottleNeck(in_channels//2, ) for _ in range(BottleNeck_repeats)],
        )

        self.group_conv3 = nn.Conv2d(4*in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.group_conv4 = nn.Conv2d(4*in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_expanded1 = self.group_conv1(x)
        x_expanded2 = self.group_conv2(x)
        x_expanded = torch.concat(x_expanded1, x_expanded2, dim=1)

        x_processed1 = self.conv1(x_expanded)
        x_processed1 = self.conv2(x_processed1)

        x_processed2 = self.conv3(x_processed1)
        x_processed2 = self.conv4(x_processed2)

        x_expanded_part1, x_expanded_part2 = torch.split(x_expanded, x_expanded.shape[1] // 2, dim=1)
        x_processed1_part1, x_processed1_part2 = torch.split(x_processed1, x_processed1.shape[1] // 2, dim=1)
        x_processed2_part1, x_processed2_part2 = torch.split(x_processed2, x_processed2.shape[1] // 2, dim=1)

        shuffle_part1 = torch.concat(x_expanded_part1, x_expanded_part1, x_processed1_part1, x_processed2_part1)
        shuffle_part2 = torch.concat(x_expanded_part2, x_expanded_part2, x_processed1_part2, x_processed2_part2)
        
        x_merged = torch.concat(self.group_conv3(shuffle_part1), self.group_conv4(shuffle_part2), dim=1)
        
        return self.conv5(x_merged)

class EELAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4):
        super(EELAN, self).__init__()

        self.blocks = nn.Sequential(*[EELANBlock(in_channels, out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

# Example usage:

model = EELAN(3, 64)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y.shape)