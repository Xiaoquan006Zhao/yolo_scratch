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
from v9_scratch.Blocks.ScaledPredictions import (
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
        self.layer_outputs = []

        self.inference_layers = nn.ModuleList([ 
            Silence(),
            Conv(self.in_channels, 64, k=3, s=2), 
            Conv(64, 128, k=3, s=2), 
            RepNCSPELAN4(256, 128, 64, 1),

            Conv(128, 256, k=3, s=2), 
            RepNCSPELAN4(512, 256, 128, 1),

            Conv(256, 512, k=3, s=2), 
            RepNCSPELAN4(512, 512, 256, 1),

            Conv(512, 512, k=3, s=2), 
            RepNCSPELAN4(512, 512, 256, 1),
            
            # --------------------------
            SPPELAN(512, 512, 256),

            Upsample(),
            Concat([7, -1], 1),
            RepNCSPELAN4(512, 512, 256, 1),

            Upsample(),
            Concat([5, -1], 1),
            RepNCSPELAN4(256, 256, 128, 1),

            Downsample(256),
            Concat([13, -1], 1),
            RepNCSPELAN4(512, 512, 256, 1),

            Downsample(512),
            Concat([10, -1], 1),
            RepNCSPELAN4(512, 512, 256, 1),
        ]) 

        self.inference_prediction = nn.ModuleList([
            ScaledPredictions([16, 19, 22], self.num_classes)
        ])

        self.auxiliary_layers = nn.ModuleList([ 
            CBLinear(5, [[256]]),
            CBLinear(7, [[256, 512]]),
            CBLinear(9, [[256, 512, 512]]),

            Conv(self.in_channels, 64, k=3, s=2), 
            Conv(64, 128, k=3, s=2), 
            RepNCSPELAN4(256, 128, 64, 1),

            Conv(128, 256, k=3, s=2), 

            CBFuse([23, 24, 25, -1], ),
            RepNCSPELAN4(512, 256, 128, 1),

            Conv(256, 512, k=3, s=2), 
            CBFuse([24, 25, -1]),
            RepNCSPELAN4(512, 512, 256, 1),

            Conv(512, 512, k=3, s=2), 
            CBFuse([25, -1]),
            RepNCSPELAN4(512, 512, 256, 1),

            ScaledPredictions([31, 34, 37, 16, 19, 22], self.num_classes)
        ]) 
    
    def forward(self, x): 
        if self.TRAINING:
            self.layers = self.inference_layers + self.auxiliary_layers
        else:
            self.layers = self.inference_layers + self.inference_prediction

        for layer in self.layers:
            if isinstance(layer, ScaledPredictions):
                predictions = layer(self.layer_outputs)
                return predictions
            if isinstance(layer, (CBFuse, CBLinear, Concat)):
                x = layer(self.layer_outputs)
            else:
                x = layer(x)
            self.layer_outputs.append(x)
