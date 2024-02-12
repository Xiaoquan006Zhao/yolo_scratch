import torch
import torch.nn as nn

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))

class SPP(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes
        ])

    def forward(self, x):
        features = [x] + [maxpool(x) for maxpool in self.maxpools]
        return torch.cat(features, 1)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super(ResidualBlock, self).__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self.block = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, 1),
            ConvBlock(hidden_channels, out_channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.reduce = ConvBlock(in_channels, out_channels, 1)
        self.split_conv = ConvBlock(out_channels, out_channels // 2, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels // 2, out_channels // 2) for _ in range(num_blocks)])
        self.merge_conv = ConvBlock(out_channels // 2, out_channels // 2, 1)

    def forward(self, x):
        x = self.reduce(x)
        x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
        x1 = self.split_conv(x1)
        x2 = self.blocks(x2)
        x2 = self.merge_conv(x2)
        return torch.cat([x1, x2], dim=1)

class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.stem = ConvBNMish(3, 32, 3)  # Example initial layer
        # Define CSPDarknet53 structure here, including CSPBlocks and any transition layers
        # Placeholder for CSPBlock and other layers
        self.stage1 = CSPBlock(32, 64, num_blocks=1)  # Example for stage1, adjust num_blocks and channels as necessary

        # Continue defining CSPBlocks for other stages

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        # Forward through remaining stages
        return x

# Example usage
model = CSPDarknet53()
print(model)
