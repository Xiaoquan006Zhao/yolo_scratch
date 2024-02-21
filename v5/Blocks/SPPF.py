import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph

from .BasicBlock import ConvBNMish

class SPPFBlock(nn.Module):
    def __init__(self, in_channels, pool_size, pool_repeats):
        super(SPPFBlock, self).__init__()
        self.pool_repeats = pool_repeats
        self.pool_size = pool_size
        self.conv1 = ConvBNMish(in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.conv_out = ConvBNMish(in_channels * (pool_repeats + 1), in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        features = [x]  # Start with the original features
        pool = x

        for _ in range(self.pool_repeats):
            pool = F.max_pool2d(pool, kernel_size=self.pool_size, stride=1, padding=2)
            features.append(pool)
        
        # Concatenate along the channel dimension, (dim=0 is batch dimension)
        features = torch.cat(features, dim=1)
        
        return self.conv_out(features)

# test_shape = (1,512,13,13)
# x = torch.randn(test_shape)

# def test_SPPBlock():
#     model = SPPBlock()
#     print(model(x).shape)
#     # print(model)
#     # del model
#     return model

# model = test_SPPBlock()

# architecture = 'denselayer'
# model_graph = draw_graph(model, input_size=(test_shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
# model_graph.visual_graph