import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph
from denseBlock import *

class SPPBlock(nn.Module):
    def __init__(self, pool_sizes=[1, 5, 9, 13]):
        super(SPPBlock, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        # x is the input feature map, assumed to be of size [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        output = [x]  # Include the original feature map in the output
        
        for pool_size in self.pool_sizes:
            # Calculate kernel and stride sizes to cover the whole feature map
            kernel_size = (height // pool_size, width // pool_size)
            stride = kernel_size
            
            # Apply max pooling
            pooled = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)
            
            # Upsample to the original feature map size
            upsampled = F.interpolate(pooled, size=(height, width), mode='bilinear', align_corners=False)
            
            # Collect the upsampled feature maps
            output.append(upsampled)
        
        # Concatenate along the channel dimension
        output = torch.cat(output, 1)  # New shape: [batch_size, channels * (len(pool_sizes) + 1), height, width]
        return output

test_shape = (1,512,13,13)
x = torch.randn(test_shape)

def test_SPPBlock():
    model = SPPBlock()
    print(model(x).shape)
    # print(model)
    # del model
    return model

model = test_SPPBlock()

architecture = 'denselayer'
model_graph = draw_graph(model, input_size=(test_shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
model_graph.visual_graph