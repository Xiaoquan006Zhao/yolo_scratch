import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph

from .BottleNeck import BottleNeck
from .BasicBlock import ConvBNMish
from torchvision.ops import drop_block2d

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleNeck_use_residual, BottleNeck_repeats, use_dropblock=False, dropblock_params={'block_size': 5, 'p': 0.1}):
        super(CSPBlock, self).__init__()
        self.process_block = BottleNeck(in_channels//2, bottleNeck_use_residual)
        self.BottleNeck_repeats = BottleNeck_repeats
        self.use_dropblock = use_dropblock
        self.dropblock_params = dropblock_params

        self.in_channels = in_channels

        # Halved due to CSP split
        conv_input_channels = in_channels // 2
        self.conv_part1 = ConvBNMish(conv_input_channels, conv_input_channels, kernel_size=1, stride=1, padding=0)
        self.conv_part2 = ConvBNMish(conv_input_channels, conv_input_channels, kernel_size=1, stride=1, padding=0)

        self.conv_out = ConvBNMish(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Split input into two parts
        part1, part2 = torch.split(x, x.shape[1] // 2, dim=1)  # Split channels into two halves
        
        part1 = self.conv_part1(part1)
        part2 = self.conv_part2(part2)

        # Process part1 through transition (if necessary) and part2 through the specified process_block
        for _ in range(self.BottleNeck_repeats):
            part2 = self.process_block(part2)
        
        # Concatenate the processed parts along dimension 1 (channel dimension)
        # dimension 0 is Batch dimension
        out = torch.cat((part1, part2), dim=1)

        # if self.use_dropblock:
        #     # Apply DropBlock on the concatenated output
        #     out = drop_block2d(out, **self.dropblock_params)

        out = self.conv_out(out)

        return out
    
# test_shape = (1,3,224,224)
# x = torch.randn(test_shape)
# conv = nn.Conv2d(3, 8, 3)
# x = conv(x)
    
# def test_CSPBlock():
#     model = DenseBlock(5,x.shape[1])
#     dense_block = DenseBlock(layer_num=4, in_channels=4)  # Example DenseBlock initialization
#     model = CSPBlock(process_block=dense_block, in_channels=8)

#     print('CSPblock Output shape : ',model(x).shape)
#     print('Model ',model)
#     # del model
#     return model

# model = test_CSPBlock()

# architecture = 'denseblock'
# model_graph = draw_graph(model, input_size=(x.shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
# model_graph.visual_graph