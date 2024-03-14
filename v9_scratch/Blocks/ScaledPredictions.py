import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicBlock import ConvBNMish

class ScaledPrediction(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super(ScaledPrediction, self).__init__()
        self.num_classes = num_classes
        out_channels = (num_classes + 5) * 3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    # Defining the forward pass and reshaping the output to the desired output 
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
    def forward(self, x): 
        output = self.conv1(x) 
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output = output.permute(0, 1, 3, 4, 2) 
        return output

class ScaledPredictions(nn.Module):
    def __init__(self, idx, num_classes): 
        super(ScaledPredictions, self).__init__()
        self.num_classes = num_classes
        self.out_channels = (num_classes + 5) * 3
        self.idx = idx
        
    def forward(self, xs): 
        # Sort idx based on increasing channel size
        sorted_idx = sorted(self.idx, key=lambda i: xs[i].shape[1])

        convs = []
        outputs = []

        for i in sorted_idx:
            convs.append(nn.Conv2d(xs[i].shape[1], self.out_channels, kernel_size=1, stride=1, padding=0))

        for conv, i in zip(convs, sorted_idx):
            x = xs[i]
            output = conv(x) 
            output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
            output = output.permute(0, 1, 3, 4, 2) 
            outputs.append(output)

        return outputs

   


