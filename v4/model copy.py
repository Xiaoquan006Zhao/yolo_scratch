import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACTIVATIONS = {
    'mish': Mish(),
    'linear': nn.Identity()
}

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='mish'):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        return self.activation(self.conv(x))

class DenseCSPBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseCSPBlock, self).__init__()
        # Assuming growth_rate is the increase in channels for each block
        self.conv1x1 = Conv(in_channels, growth_rate, 1)
        self.conv3x3 = Conv(growth_rate, growth_rate, 3)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.conv3x3(out)
        return torch.cat([x, out], 1)  # Concatenate features from input and output

class CSPStageDense(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, growth_rate):
        super(CSPStageDense, self).__init__()

        self.downsample_conv = Conv(in_channels, out_channels, 3, stride=2)

        # Splitting into two paths, only one of which will be DenseNet-like
        self.split_conv0 = Conv(out_channels, out_channels // 2, 1)
        self.split_conv1 = Conv(out_channels, out_channels // 2, 1)

        self.blocks_conv = nn.Sequential(*[DenseCSPBlock(out_channels // 2 + i * growth_rate, growth_rate) for i in range(num_blocks)])

        self.final_conv = Conv(out_channels // 2 + num_blocks * growth_rate, out_channels // 2, 1)

        self.concat_conv = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)
        x1 = self.final_conv(x1)  # Condensing the expanded channels back to original size

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

class CSPDarknet53(nn.Module):
    def __init__(self, stem_channels=32, feature_channels=[64, 128, 256, 512, 1024], num_features=1):
        super(CSPDarknet53, self).__init__()

        self.stem_conv = Conv(3, stem_channels, 3)

        self.stages = nn.ModuleList([
            CSPFirstStage(stem_channels, feature_channels[0]),
            CSPStage(feature_channels[0], feature_channels[1], 2),
            CSPStage(feature_channels[1], feature_channels[2], 8),
            CSPStage(feature_channels[2], feature_channels[3], 8),
            CSPStage(feature_channels[3], feature_channels[4], 4)
        ])
 
        self.feature_channels = feature_channels
        self.num_features = num_features

    def forward(self, x):
        x = self.stem_conv(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features[-self.num_features:]

def _BuildCSPDarknet53(num_features=3):
    model = CSPDarknet53(num_features=num_features)

    return model, model.feature_channels[-num_features:]

if __name__ == '__main__':
    model = CSPDarknet53()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)