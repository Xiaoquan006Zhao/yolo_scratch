import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicBlock import (
    autopad,
)

class CBLinear(nn.Module):
    def __init__(self, route_list, in_channels, c2s, k=1, s=1, p=None, g=1):
        super(CBLinear, self).__init__()
        self.route_list = route_list
        self.c2s = c2s
        self.conv = nn.Conv2d(in_channels, sum(self.c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, xs):
        x = xs[0]
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    def __init__(self, route_list, idx):
        super(CBFuse, self).__init__()
        self.idx = idx
        self.route_list = route_list

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out
