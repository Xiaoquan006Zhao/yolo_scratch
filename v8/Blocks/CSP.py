import torch
import torch.nn as nn
from .BottleNeck import BottleNeck
from .BasicBlock import ConvBNMish
from torchvision.ops import drop_block2d

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleNeck_use_residual, BottleNeck_repeats, use_dropblock=True, dropblock_params={'block_size': 3, 'p': 0.1}):
        super(CSPBlock, self).__init__()

        self.process_blocks = nn.ModuleList(
            [BottleNeck(out_channels//2, bottleNeck_use_residual) for _ in range(BottleNeck_repeats)],
        )

        self.BottleNeck_repeats = BottleNeck_repeats
        self.use_dropblock = use_dropblock
        self.dropblock_params = dropblock_params

        self.in_channels = in_channels

        self.conv1 = ConvBNMish(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNMish((BottleNeck_repeats+2)*out_channels//2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        concate_list = []

        part1, part2 = torch.split(x, x.shape[1] // 2, dim=1)  # Split channels into two halves
        concate_list.append(part1)
        concate_list.append(part2)

        for process_block in self.process_blocks:
            part2 = process_block(part2)
            concate_list.append(part2)

        out = torch.cat(concate_list, dim=1)

        if self.use_dropblock:
            out = drop_block2d(out, **self.dropblock_params)

        return self.conv2(out)
