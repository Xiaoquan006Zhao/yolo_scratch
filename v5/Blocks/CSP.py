import torch
import torch.nn as nn
from .BottleNeck import BottleNeck
from .BasicBlock import ConvBNMish
from torchvision.ops import drop_block2d

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleNeck_use_residual, BottleNeck_repeats, use_dropblock=True, dropblock_params={'block_size': 3, 'p': 0.1}):
        super(CSPBlock, self).__init__()

        self.process_blocks = nn.Sequential(
            *[BottleNeck(in_channels//2, bottleNeck_use_residual) for _ in range(BottleNeck_repeats)],
        )
        self.in_channels = in_channels

        self.BottleNeck_repeats = BottleNeck_repeats
        self.use_dropblock = use_dropblock
        self.dropblock_params = dropblock_params

        conv_input_channels = in_channels // 2
        self.conv_part1 = ConvBNMish(conv_input_channels, conv_input_channels, kernel_size=1, stride=1, padding=0)
        self.conv_part2 = ConvBNMish(conv_input_channels, conv_input_channels, kernel_size=1, stride=1, padding=0)

        self.conv_out = ConvBNMish(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        part1, part2 = torch.split(x, x.shape[1] // 2, dim=1)  # Split channels into two halves
        part1 = self.conv_part1(part1)
        part2 = self.conv_part2(part2)
        part2 = self.process_blocks(part2)
        out = torch.cat((part1, part2), dim=1)

        # if self.use_dropblock:
        #     out = drop_block2d(out, **self.dropblock_params)

        out = self.conv_out(out)

        return out
    