import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock(nn.Module):
    def __init__(self, block_size, keep_probability):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_probability
        self.gamma = None

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = F.max_pool2d(mask.unsqueeze(1), kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
            block_mask = 1 - block_mask.squeeze(1)

            # Recompute keep_prob
            out = x * block_mask * (block_mask.numel() / block_mask.sum())  
            return out

    def _compute_gamma(self, x):
        b, c, h, w = x.size()
        return self.keep_prob / (self.block_size ** 2) * ((h * w) / ((h - self.block_size + 1) * (w - self.block_size + 1)))
