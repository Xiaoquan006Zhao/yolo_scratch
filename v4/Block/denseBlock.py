import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph
import config

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_parameters={}
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32]
model_parameters['densenet264'] = [6,12,64,48]

# growth rate
k = config.dense_growth_rate
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

        return x

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

class DenseBlock(nn.Module):
    def __init__(self, layer_num, in_channels):
        """
        Looping through total number of layers in the denseblock. 
        Adding k number of channels in each loop as each layer generates tensor with k channels.
        Args:
            layer_num (int) : total number of dense layers in the dense block
            in_channels (int) : input number of channels 
        """
        super(DenseBlock,self).__init__()
        self.layer_num = layer_num
        self.deep_nn = nn.ModuleList()

        for num in range(self.layer_num):
            self.deep_nn.add_module(f"DenseLayer_{num}", DenseLayer(in_channels+k*num))

    def forward(self, x):
        saved_outputs = [x]  # Initialize list to save the output of each layer
        for layer in self.deep_nn:
            layer_input = torch.cat(saved_outputs, dim=1)  # Concatenate all previous outputs
            x = layer(layer_input)
            saved_outputs.append(x)  # Save current output for concatenation with future inputs

        return torch.cat(saved_outputs, dim=1) 

# test_shape = (1,3,224,224)
# def test_DenseBlock():
#     x = torch.randn(test_shape)
#     model = DenseBlock(5,3)
#     print('Denseblock Output shape : ',model(x).shape)
#     print('Model ',model)
#     # del model
#     return model

# model = test_DenseBlock()

# architecture = 'denseblock'
# model_graph = draw_graph(model, input_size=(test_shape), graph_dir ='TB' , roll=True, expand_nested=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
# model_graph.visual_graph


# class TransitionLayer(nn.Module):
#     def __init__(self, in_channels, compression_factor):
#         """
#         1x1 conv used to change output channels using the compression_factor (default = 0.5).
#         avgpool used to downsample the feature map resolution 
#         Args:
#             compression_factor (float) : output_channels/input_channels
#             in_channels (int) : input number of channels 
#         """
#         super(TransitionLayer,self).__init__()
#         self.BN = nn.BatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = int(in_channels*compression_factor) ,kernel_size = 1 ,stride = 1 ,padding = 0, bias=False)
#         self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

#     def forward(self,x):
#         """
#         Args:
#             x (tensor) : input tensor to be passed through the dense block
#         Attributes:
#             x (tensor) : output tensor
#         """
#         x = self.BN(x)
#         x = self.conv1(x)
#         x = self.avgpool(x)
#         return x

# class DenseNet(nn.Module):
#     def __init__(self, densenet_variant, in_channels, num_classes=1000):
#         """
#         Creating an initial 7x7 convolution followed by 3 DenseBlock and 3 Transition layers. Concluding this with 4th DenseBlock, 7x7 global average pool and FC layer
#         for classification  
#         Args:
#             densenet_variant (list) : list containing the total number of layers in a dense block
#             in_channels (int) : input number of channels
#             num_classes (int) : Total nnumber of output classes 
#         """
#         super(DenseNet,self).__init__()
#         # 7x7 conv with s=2 and maxpool
#         self.conv1 = nn.Conv2d(in_channels=in_channels ,out_channels=64 ,kernel_size=7 ,stride=2 ,padding=3 ,bias = False)
#         self.BN1 = nn.BatchNorm2d(num_features=64)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         # adding 3 DenseBlocks and 3 Transition Layers 
#         self.deep_nn = nn.ModuleList()
#         dense_block_inchannels = 64

#         for num in range(len(densenet_variant))[:-1]:
#             self.deep_nn.add_module( f"DenseBlock_{num+1}" , DenseBlock( densenet_variant[num] , dense_block_inchannels ) )
#             dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[num])

#             self.deep_nn.add_module( f"TransitionLayer_{num+1}" , TransitionLayer( dense_block_inchannels,compression_factor ) )
#             dense_block_inchannels = int(dense_block_inchannels*compression_factor)

#         # adding the 4th and final DenseBlock
#         self.deep_nn.add_module( f"DenseBlock_{num+2}" , DenseBlock( densenet_variant[-1] , dense_block_inchannels ) )
#         dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[-1])

#         self.BN2 = nn.BatchNorm2d(num_features=dense_block_inchannels)

#         # Average Pool
#         self.average_pool = nn.AdaptiveAvgPool2d(1)
        
#         # fully connected layer
#         self.fc1 = nn.Linear(dense_block_inchannels, num_classes)


#     def forward(self,x):
#         """
#         deep_nn is the module_list container which has all the dense blocks and transition blocks
#         """
#         x = self.relu(self.BN1(self.conv1(x)))
#         x = self.maxpool(x)
        
#         for layer in self.deep_nn:
#             x = layer(x)
            
#         x = self.relu(self.BN2(x))
#         x = self.average_pool(x)

#         x = torch.flatten(x, start_dim=1)
#         x = self.fc1(x)

#         # print(x.shape)
#         return x

# x = torch.randn(1,3,224,224)
# model = DenseNet(model_parameters['densenet121'],3)

# architecture = 'denseNet'
# model_graph = draw_graph(model, input_size=(1,3,224,224), graph_dir ='TB' , roll=False, expand_nested=True, show_shapes=True, graph_name=f'self_{architecture}',save_graph=True,filename=f'self_{architecture}')
# model_graph.visual_graph


