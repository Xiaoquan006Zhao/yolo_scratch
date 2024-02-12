import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_parameters={}
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32]
model_parameters['densenet264'] = [6,12,64,48]

# growth rate
k = 32
compression_factor = 0.5

class DenseLayer(nn.Module):
    def __init__(self,in_channels):
        """
        First 1x1 convolution generating 4*k number of channels irrespective of the total number of input channels.
        First 3x3 convolution generating k number of channels from the 4*k number of input channels.
        Args:
        in_channels (int) : # input channels to the Dense Layer
        """
        super(DenseLayer,self).__init__()
        self.BN1 = nn.BatchNorm2d(num_features = in_channels)
        self.conv1 = nn.Conv2d( in_channels=in_channels , out_channels=4*k , kernel_size=1 , stride=1 , padding=0 , bias = False )
        self.BN2 = nn.BatchNorm2d(num_features = 4*k)
        self.conv2 = nn.Conv2d( in_channels=4*k , out_channels=k , kernel_size=3 , stride=1 , padding=1 , bias = False )
        self.relu = nn.ReLU()

    def forward(self,x):
        """
        Bottleneck DenseLayer with following operations
        (i) batchnorm -> relu -> 1x1 conv
        (ii) batchnorm -> relu -> 3x3 conv
        Concatenation of input and output tensor which is the main idea of DenseNet. 
        Args:
            x (tensor) : input tensor to be passed through the dense layer
        Attributes:
            x (tensor) : output tensor 
        """
        x_in = x
        # BN -> relu -> conv(1x1)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv1(x)
        # BN -> relu -> conv(3x3)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.cat([x_in,x],1)

        return x

def test_DenseLayer():
    x = torch.randn(1,64,224,224)
    model = DenseLayer(64)
    # print(model(x).shape)
    # print(model)
    # del model
    return model


model = test_DenseLayer()

architecture = 'denselayer'
model_graph = draw_graph(model, input_size=(1,64,224,224), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
model_graph.visual_graph




