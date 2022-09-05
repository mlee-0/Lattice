from typing import Tuple

import numpy as np
import torch


def get_parameter_count(model: torch.nn.Module) -> int:
    """Get the number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LatticeCnn(torch.nn.Module):
    """3D CNN whose input is a 3D array of densities and whose output is a 1D array of strut diameters."""

    def __init__(self) -> None:
        super().__init__()

        h, w, d = 51, 51, 51
        input_channels = 1
        strut_neighborhood = (3, 3, 3)

        # Number of output channels in the first layer.
        c = 2

        # Parameters for all convolution layers.
        k = 5
        s = 2
        p = 1

        # Total number of nodes in input image.
        n = h * w * d
        # Output size, representing the maximum number of struts possible.
        size_output = np.prod((
            np.prod([dimension - (strut_neighborhood[i] - 1) for i, dimension in enumerate((h, w, d))]),
            np.prod(strut_neighborhood) - 1,
        ))

        self.convolution_1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=k, stride=s, padding=p),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*1, out_channels=c*2, kernel_size=k, stride=s, padding=p),
            torch.nn.BatchNorm3d(c*2),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*2, out_channels=c*4, kernel_size=k, stride=s, padding=p),
            torch.nn.BatchNorm3d(c*4),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_4 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*4, out_channels=c*8, kernel_size=k, stride=s, padding=p),
            torch.nn.BatchNorm3d(c*8),
            torch.nn.ReLU(inplace=True),
        )

        # Size of output of all convolution layers.
        shape_convolution = self.convolution_4(self.convolution_3(self.convolution_2(self.convolution_1(torch.empty((1, input_channels, h, w, d)))))).size()
        size_convolution = np.prod(shape_convolution[1:])
        print(f'linear: {(size_convolution, size_output)}')

        self.linear = torch.nn.Linear(in_features=size_convolution, out_features=size_output)

    def forward(self, x):
        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.linear(x.flatten())
        return x

class NodeCnn(torch.nn.Module):
    """3D CNN whose input is a 3D array of densities and whose output is a 3D array of node locations with the same shape as the input."""

    def __init__(self) -> None:
        super().__init__()

        h, w, d = 51, 51, 51
        input_channels = 1
        output_channels = 1

        # Number of output channels in the first layer.
        c = 4

        # Parameters for all convolution layers.
        k = 5
        s = 2
        p = 1

        self.convolution_1 = torch.nn.Sequential(
            torch.nn.Conv3d(input_channels, c*1, kernel_size=k, stride=s, padding=p, padding_mode="zeros"),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_2 = torch.nn.Sequential(
            torch.nn.Conv3d(c*1, c*2, kernel_size=k, stride=s, padding=p, padding_mode="zeros"),
            torch.nn.BatchNorm3d(c*2),
            torch.nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_3 = torch.nn.Sequential(
            torch.nn.Conv3d(c*2, c*4, kernel_size=k, stride=s, padding=p, padding_mode="zeros"),
            torch.nn.BatchNorm3d(c*4),
            torch.nn.ReLU(inplace=True),
        )

        # Convenience functions for returning residual blocks and squeeze-and-excitation blocks.
        residual_block = lambda: torch.nn.Sequential(
            torch.nn.Conv3d(c*4, c*4, kernel_size=3, padding="same"),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(c*4),
            torch.nn.Conv3d(c*4, c*4, kernel_size=3, padding="same"),
            torch.nn.BatchNorm3d(c*4),
        )
        se_block = lambda kernel_size: torch.nn.Sequential(
            torch.nn.AvgPool3d(kernel_size=kernel_size),
            torch.nn.Flatten(),
            torch.nn.Linear(c*4, c*4//16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(c*4//16, c*4),
            torch.nn.Sigmoid(),
        )
        
        output_size_residual =  self.convolution_3(
                                self.convolution_2(
                                self.convolution_1(
                                    torch.empty((1, input_channels, h, w, d))
                                ))).size()[2:]
        self.residual_1 = residual_block()
        self.se_1 = se_block(output_size_residual)
        self.residual_2 = residual_block()
        self.se_2 = se_block(output_size_residual)
        self.residual_3 = residual_block()
        self.se_3 = se_block(output_size_residual)
        self.residual_4 = residual_block()
        self.se_4 = se_block(output_size_residual)
        self.residual_5 = residual_block()
        self.se_5 = se_block(output_size_residual)

        self.deconvolution_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(c*4, c*2, kernel_size=k, stride=s, padding=p, output_padding=1, padding_mode="zeros"),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(c*2),
        )
        self.deconvolution_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(c*2, c*1, kernel_size=k, stride=s, padding=p, output_padding=0, padding_mode="zeros"),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(c*1),
        )
        self.deconvolution_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(c*1, output_channels, kernel_size=k, stride=s, padding=p, output_padding=0, padding_mode="zeros"),
            # torch.nn.BatchNorm3d(OUTPUT_CHANNELS, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)

        residual = self.residual_1(x)
        se = self.se_1(x)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_2(x)
        se = self.se_2(x)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_3(x)
        se = self.se_3(x)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_4(x)
        se = self.se_4(x)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_5(x)
        se = self.se_5(x)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))

        x = self.deconvolution_1(x)
        x = self.deconvolution_2(x)
        x = self.deconvolution_3(x)

        return x

class NodeStrutCnn(torch.nn.Module):
    """3D CNN whose input is a 3D array of densities and whose output is a 2-channel 3D array of node locations and strut diameters."""

    def __init__(self) -> None:
        super().__init__()

        h, w, d = 51, 51, 51
        input_channels = 1
        output_channels = 1

        # Number of output channels in the first layer.
        c = 32

        # Parameters for all convolution layers.
        k = (4, 4, 4)
        s = (2, 2, 2)
        p = 1

        self.convolution_1 = torch.nn.Sequential(
            torch.nn.Conv3d(input_channels, c*1, kernel_size=9, stride=1, padding="same", padding_mode="zeros"),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_2 = torch.nn.Sequential(
            torch.nn.Conv3d(c*1, c*2, kernel_size=k, stride=s, padding=p, padding_mode="zeros"),
            torch.nn.BatchNorm3d(c*2),
            torch.nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_3 = torch.nn.Sequential(
            torch.nn.Conv3d(c*2, c*4, kernel_size=k, stride=s, padding=p, padding_mode="zeros"),
            torch.nn.BatchNorm3d(c*4),
            torch.nn.ReLU(inplace=True),
        )

        # Convenience functions for returning residual blocks and squeeze-and-excitation blocks.
        residual_block = lambda: torch.nn.Sequential(
            torch.nn.Conv3d(c*4, c*4, kernel_size=3, padding="same"),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(c*4),
            torch.nn.Conv3d(c*4, c*4, kernel_size=3, padding="same"),
            torch.nn.BatchNorm3d(c*4),
        )
        se_block = lambda kernel_size: torch.nn.Sequential(
            torch.nn.AvgPool3d(kernel_size=kernel_size),
            torch.nn.Flatten(),
            torch.nn.Linear(c*4, c*4//16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(c*4//16, c*4),
            torch.nn.Sigmoid(),
        )
        
        output_size_residual =  self.convolution_3(
                                self.convolution_2(
                                self.convolution_1(
                                    torch.empty((1, input_channels, h, w, d))
                                ))).size()[2:]
        self.residual_1 = residual_block()
        self.se_1 = se_block(output_size_residual)
        self.residual_2 = residual_block()
        self.se_2 = se_block(output_size_residual)
        self.residual_3 = residual_block()
        self.se_3 = se_block(output_size_residual)
        self.residual_4 = residual_block()
        self.se_4 = se_block(output_size_residual)
        self.residual_5 = residual_block()
        self.se_5 = se_block(output_size_residual)

        self.deconvolution_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(c*4, c*2, kernel_size=k, stride=s, padding=p, output_padding=(0,0), padding_mode="zeros"),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(c*2),
        )
        self.deconvolution_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(c*2, c*1, kernel_size=k, stride=s, padding=p, output_padding=(0,0), padding_mode="zeros"),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(c*1),
        )
        self.deconvolution_3 = torch.nn.Sequential(
            torch.nn.Conv3d(c*1, output_channels, kernel_size=9, stride=1, padding="same", padding_mode="zeros"),
            # torch.nn.BatchNorm3d(OUTPUT_CHANNELS, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            torch.nn.Sigmoid(),
            # torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)

        x = self.residual_1(x)
        x = self.se_1(x)
        x = self.residual_2(x)
        x = self.se_2(x)
        x = self.residual_3(x)
        x = self.se_3(x)
        x = self.residual_4(x)
        x = self.se_4(x)
        x = self.residual_5(x)
        x = self.se_5(x)

        x = self.deconvolution_1(x)
        x = self.deconvolution_2(x)
        x = self.deconvolution_3(x)

        return x


class LatticeGnn(torch.nn.Module):
    """A GNN whose input is a fully-connected graph of node locations and whose output is a graph of strut diameters."""

    # from torch_geometric.nn import GCNConv

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x