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
    ScaledPrediction,
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
            ConvBNMish(self.in_channels, 64, kernel_size=3, stride=2, padding=1), 
            ConvBNMish(64, 128, kernel_size=3, stride=2, padding=1), 
            ConvBNMish(128, 256, kernel_size=1, stride=1, padding=0), 
            ConvBNMish(256, 256, kernel_size=3, stride=2, padding=1), 
            ScaledPrediction(256, self.num_classes),
            ConvBNMish(256, 512, kernel_size=1, stride=1, padding=0), 
            ConvBNMish(512, 512, kernel_size=3, stride=2, padding=1), 
            ScaledPrediction(512, self.num_classes),
            ConvBNMish(512, 512, kernel_size=1, stride=1, padding=0), 
            ConvBNMish(512, 512, kernel_size=3, stride=2, padding=1), 
            ConvBNMish(512, 512, kernel_size=1, stride=1, padding=0),  
            ScaledPrediction(512, self.num_classes),
        ]) 
       
    def forward(self, x): 
        outputs = []
        for layer in self.inference_layers:
            if isinstance(layer, ScaledPrediction):
                prediction = layer(x)
                outputs.append(prediction)
            else:
                x = layer(x)

            self.layer_outputs.append(x)
        return outputs[::-1]
        
