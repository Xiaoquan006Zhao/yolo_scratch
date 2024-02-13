import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph

from denseBlock import *


class CSPBlock(nn.Module):
    def __init__(self, process_block, in_channels):
        """
        Initialize the CSPBlock.
        Args:
        dense_block (DenseBlock): The dense block to be used in one of the paths.
        in_channels (int): Number of input channels to the block.
        transition_channels (int, optional): Number of channels after transition layer, if None, it uses in_channels.
        """
        super(CSPBlock, self).__init__()
        self.process_block = process_block

    def forward(self, x):
        """
        Forward pass of the CSPBlock.
        Args:
        x (tensor): Input tensor to the block.
        Returns:
        tensor: Output tensor of the CSPBlock.
        """
        # Split input into two parts
        part1, part2 = torch.split(x, x.shape[1] // 2, dim=1)  # Split channels into two halves
        
        # Process part1 through transition (if necessary) and part2 through DenseBlock
        part2_dense = self.process_block(part2)
        
        # Concatenate the processed parts
        out = torch.cat((part1, part2_dense), dim=1)
        return out

test_shape = (1,3,224,224)
def test_CSPBlock():
    x = torch.randn(test_shape)
    model = DenseBlock(5,3)
    dense_block = DenseBlock(layer_num=4, in_channels=4)  # Example DenseBlock initialization
    model = CSPBlock(process_block=dense_block, in_channels=8)

    print('CSPblock Output shape : ',model(x).shape)
    print('Model ',model)
    # del model
    return model

model = test_CSPBlock()

architecture = 'denseblock'
model_graph = draw_graph(model, input_size=(test_shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
model_graph.visual_graph