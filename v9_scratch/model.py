import torch.nn as nn 
import config
from Blocks.BasicBlock import (
    ConvBNMish,
    Conv,
    Silence,
    Upsample,
    Downsample,
    Concat,
)
from Blocks.RepNCSPELAN4 import (
    RepNCSPELAN4,
)
from Blocks.ScaledPredictions import (
    ScaledPredictions,
)
from Blocks.CB import (
    CBLinear,
    CBFuse,
)
from Blocks.SPPELAN import (
    SPPELAN,
)

class YOLOv9(nn.Module): 
    def __init__(self, in_channels=3, num_classes=20, TRAINING=True): 
        super().__init__() 
        self.TRAINING = TRAINING
        self.num_classes = num_classes 
        self.in_channels = in_channels 
        

        inference_layers = nn.ModuleList([ 
            Silence(),
            Conv(self.in_channels, 64, k=3, s=2), 
            Conv(64, 128, k=3, s=2), 
            RepNCSPELAN4(128, 256, 128, 64),

            Conv(256, 256, k=3, s=2), 
            RepNCSPELAN4(256, 512, 256, 128),

            Conv(512, 512, k=3, s=2), 
            RepNCSPELAN4(512, 512, 512, 256),

            Conv(512, 512, k=3, s=2), 
            RepNCSPELAN4(512, 512, 512, 256),
            
            # --------------------------
            SPPELAN(512, 512, 256),

            Upsample(),
            Concat([7, -1], 1),
            RepNCSPELAN4(1024, 512, 512, 256),

            Upsample(),
            Concat([5, -1], 1),
            RepNCSPELAN4(1024, 256, 256, 128),

            Downsample(256),
            Concat([13, -1], 1),
            RepNCSPELAN4(768, 512, 512, 256),

            Downsample(512),
            Concat([10, -1], 1),
            RepNCSPELAN4(1024, 512, 512, 256),

            CBLinear([5], 512, [256]),
            CBLinear([7], 512, [256, 512]),
            CBLinear([9], 512, [256, 512, 512]),

            Conv(self.in_channels, 64, k=3, s=2), 
            Conv(64, 128, k=3, s=2), 
            RepNCSPELAN4(128, 256, 128, 64),

            Conv(256, 256, k=3, s=2), 

            CBFuse([23, 24, 25, -1], [0,0,0]),
            RepNCSPELAN4(256, 512, 256, 128),

            Conv(512, 512, k=3, s=2), 
            CBFuse([24, 25, -1], [1,1]),
            RepNCSPELAN4(512, 512, 512, 256),

            Conv(512, 512, k=3, s=2), 
            CBFuse([25, -1], [2]),
            RepNCSPELAN4(512, 512, 512, 256),

            ScaledPredictions([31, 34, 37, 16, 19, 22], [512, 512, 512, 256, 512, 512], self.num_classes),
        ]) 

        inference_prediction = nn.ModuleList([
            ScaledPredictions([16, 19, 22], [256, 512, 512], self.num_classes),
        ])

        auxiliary_layers = nn.ModuleList([ 
            CBLinear([5], 512, [256]),
            CBLinear([7], 512, [256, 512]),
            CBLinear([9], 512, [256, 512, 512]),

            Conv(self.in_channels, 64, k=3, s=2), 
            Conv(64, 128, k=3, s=2), 
            RepNCSPELAN4(128, 256, 128, 64),

            Conv(256, 256, k=3, s=2), 

            CBFuse([23, 24, 25, -1], [0,0,0]),
            RepNCSPELAN4(256, 512, 256, 128),

            Conv(512, 512, k=3, s=2), 
            CBFuse([24, 25, -1], [1,1]),
            RepNCSPELAN4(512, 512, 512, 256),

            Conv(512, 512, k=3, s=2), 
            CBFuse([25, -1], [2]),
            RepNCSPELAN4(512, 512, 512, 256),

            ScaledPredictions([31, 34, 37, 16, 19, 22], [512, 512, 512, 256, 512, 512], self.num_classes),
        ]) 
    

        self.training_layers = nn.ModuleList()
        self.training_layers.extend(inference_layers)
        if self.TRAINING:
            self.training_layers.extend(auxiliary_layers)
        else:
            self.training_layers.extend(inference_prediction)

    def forward(self, x): 
        layer_outputs = []

        for layer in self.training_layers:
            if isinstance(layer, ScaledPredictions):
                route_list = layer.route_list
                selected_tensors = [layer_outputs[i] for i in route_list]

                predictions = layer(selected_tensors)
                return predictions
            elif isinstance(layer, (CBFuse, CBLinear, Concat)):
                route_list = layer.route_list
                selected_tensors = [layer_outputs[i] for i in route_list]

                x = layer(selected_tensors)
            else:
                x = layer(x)

            layer_outputs.append(x)

            if isinstance(layer, CBLinear):
                x = layer_outputs[0]

