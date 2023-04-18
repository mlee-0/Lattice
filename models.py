from typing import *

import torch
from torch.nn import *


def print_model_summary(model: Module) -> None:
    """Print information about a model."""
    print(f"\n{type(model).__name__}")
    print(f"\tTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\tLearnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

def residual(in_channels: int, out_channels: int) -> Module:
    """Return a residual block."""
    return Sequential(
        Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
        BatchNorm3d(num_features=out_channels),
        ReLU(inplace=False),
        Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
    )


class LatticeNet(Module):
    """3D CNN whose input is a 3D array of densities and whose output is an array of diameters throughout the volume."""

    def __init__(self) -> None:
        super().__init__()

        input_channels = 2
        output_channels = 13
        c = 4

        self.convolution_1 = Sequential(
            Conv3d(input_channels, c*1, kernel_size=1, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_2 = Sequential(
            Conv3d(c*1+input_channels, c*2, kernel_size=1, stride=1, padding='same'),
            BatchNorm3d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_3 = Sequential(
            Conv3d(c*2+input_channels, output_channels, kernel_size=1, stride=1, padding='same'),
            BatchNorm3d(output_channels),
            ReLU(inplace=True),
        )
        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*2, c*2)
        self.residual_3 = residual(output_channels, output_channels)

        print_model_summary(self)
    
    def forward(self, x):
        x_original = x
        x = self.convolution_1(x)
        x = torch.relu(x + self.residual_1(x))
        x = self.convolution_2(torch.cat([x_original, x], dim=1))
        x = torch.relu(x + self.residual_2(x))
        x = self.convolution_3(torch.cat([x_original, x], dim=1))
        x = torch.relu(x + self.residual_3(x))
        x = torch.clip(x, 0, 100)

        return x

class StrutNet(Module):
    """3D ResNet-based CNN whose input is a 3D array of densities and whose output is a single strut diameter."""

    def __init__(self) -> None:
        super().__init__()

        input_channels = 2

        # Number of output channels in the first layer.
        c = 4

        self.convolution_1 = Sequential(
            Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_2 = Sequential(
            Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_3 = Sequential(
            Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_4 = Sequential(
            Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_5 = Sequential(
            Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(inplace=True),
        )

        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*1, c*1)
        self.residual_3 = residual(c*1, c*1)
        self.residual_4 = residual(c*1, c*1)
        self.residual_5 = residual(c*1, c*1)

        self.pooling = AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.linear = Linear(in_features=c*1, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.convolution_1(x)
        x = torch.relu(x + self.residual_1(x))
        x = self.convolution_2(x)
        x = torch.relu(x + self.residual_2(x))
        x = self.convolution_3(x)
        x = torch.relu(x + self.residual_3(x))
        x = self.convolution_4(x)
        x = torch.relu(x + self.residual_4(x))
        x = self.convolution_5(x)
        x = torch.relu(x + self.residual_5(x))

        x = self.pooling(x)
        x = self.linear(x.view(batch_size, -1))

        return x