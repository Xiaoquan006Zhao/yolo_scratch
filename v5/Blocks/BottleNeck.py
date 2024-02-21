import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph

from .BasicBlock import ConvBNMish

class BottleNeck(nn.Module):
    def __init__(self, in_channels, use_residual = True):
        super(BottleNeck,self).__init__()
        self.conv1 = ConvBNMish(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNMish(in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.use_residual = use_residual

    def forward(self,x):
        x_residual = x
        output = self.conv2(self.conv1(x))

        return output if not self.use_residual else x_residual+output

# def test_DenseLayer():
#     x = torch.randn(1,64,224,224)
#     model = DenseLayer(64)
#     # print(model(x).shape)
#     # print(model)
#     # del model
#     return model

# model = test_DenseLayer()

# architecture = 'denselayer'
# model_graph = draw_graph(model, input_size=(1,64,224,224), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
# model_graph.visual_graph
    
