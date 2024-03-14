import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicBlock import (
    ConvBNMish,
    Conv,
)

class SPPFBlock(nn.Module):
    def __init__(self, in_channels, pool_size, pool_repeats):
        super(SPPFBlock, self).__init__()
        self.pool_repeats = pool_repeats
        self.pool_size = pool_size
        self.conv1 = ConvBNMish(in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.conv_out = ConvBNMish(in_channels //2 * (pool_repeats + 1), in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        features = [x]  # Start with the original features
        pool = x

        for _ in range(self.pool_repeats):
            pool = F.max_pool2d(pool, kernel_size=self.pool_size, stride=1, padding=0)
            features.append(F.interpolate(pool, size=x.shape[2:], mode='nearest'))
        
        features = torch.cat(features, dim=1)
        
        return self.conv_out(features)

class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
