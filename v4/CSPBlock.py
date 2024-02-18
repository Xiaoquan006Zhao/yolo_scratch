import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph
from DenseBlock import *
from torchvision.ops import drop_block2d

class CSPBlock(nn.Module):
    def __init__(self, process_block, use_dropblock=True, dropblock_params={'block_size': 5, 'p': 0.1}):
        """
        Initialize the CSPBlock.
        Args:
            process_block (nn.Module): The processing block to be used in one of the paths.
            use_dropblock (bool, optional): Flag to use DropBlock. Defaults to True.
            dropblock_params (dict, optional): Parameters for the DropBlock. 'block_size' for the size of the block to drop,
                                               and 'p' is the probability of dropping. Defaults to {'block_size': 5, 'p': 0.1}.
        """
        super(CSPBlock, self).__init__()
        self.process_block = process_block
        self.use_dropblock = use_dropblock
        self.dropblock_params = dropblock_params

        # because half of the CSP input channels is passed into process_block
        self.in_channels = process_block.in_channels * 2

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
        
        # Process part1 through transition (if necessary) and part2 through the specified process_block
        part2_processed = self.process_block(part2)
        
        # Concatenate the processed parts
        out = torch.cat((part1, part2_processed), dim=1)

        if self.use_dropblock:
            # Apply DropBlock on the concatenated output
            out = drop_block2d(out, **self.dropblock_params)

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