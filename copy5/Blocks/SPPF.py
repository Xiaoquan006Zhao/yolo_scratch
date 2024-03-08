import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicBlock import ConvBNMish

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
