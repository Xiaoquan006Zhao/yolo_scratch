import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicBlock import (
    autopad,
)

class CBLinear(nn.Module):
    def __init__(self, route_connection, c2s, k=1, s=1, p=None, g=1):
        super(CBLinear, self).__init__()
        self.route_connection = route_connection
        self.c2s = c2s
        self.k = k
        self.s = s
        self.p = p
        self.g = g

    def forward(self, xs):
        x = xs[self.route_connection]
        self.conv = nn.Conv2d(x.shape[1], sum(self.c2s), self.k, self.s, autopad(self.k, self.p), groups=self.g, bias=True)
        self.conv.weight.data = self.conv.weight.data.to(torch.float16)
        self.conv.bias.data = self.conv.bias.data.to(torch.float16)
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out
