from typing import Tuple

import numpy as np
import torch
from torch.nn import *
# import torch_geometric


def get_parameter_count(model: torch.nn.Module) -> int:
    """Get the number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function that returns a residual block.
def residual(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
        torch.nn.ReLU(inplace=False),
        torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
    )


class Cnn(torch.nn.Module):
    """3D CNN whose input is a 3D array of densities and whose output is a 1D array of strut diameters."""

    def __init__(self, device: str='cpu') -> None:
        super().__init__()

        h = w = d = 11
        input_channels = 1
        strut_neighborhood = 3
        strut_neighborhood_radius = (strut_neighborhood - 1) / 2

        self.index_channel = torch.arange(h*w*d).reshape((1, 1, h, w, d)).to('cpu').float()
        self.index_channel /= self.index_channel.max()

        # Number of output channels in the first layer.
        c = 4

        # Parameters for all convolution layers.
        k = 3
        s = 2
        p = 1

        # Output size, representing the maximum number of struts possible.
        self.shape_output = (
            16309,
        )
        size_output = int(np.prod(self.shape_output))

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
        # Convolutional layer to increase the number of channels.
        self.convolution_bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*4, out_channels=size_output, kernel_size=1, stride=1, padding='same'),
        )

        self.global_pooling = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

    def forward(self, x):
        batch_size = x.size(0)
        # x = torch.cat([x, torch.cat([self.index_channel] * batch_size, dim=0)], dim=1)

        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_bottleneck(x)

        x = self.global_pooling(x)
        # x = torch.sigmoid(x)

        x = x.reshape((batch_size, *self.shape_output))

        return x

class ResNet(torch.nn.Module):
    """3D ResNet-based CNN whose input is a 3D array of densities and whose output is a 1D array of strut diameters."""
    def __init__(self, device: str='cpu') -> None:
        super().__init__()
        self.device = device

        h = w = d = 11
        input_channels = 2
        strut_neighborhood = 3
        strut_neighborhood_radius = (strut_neighborhood - 1) / 2

        self.index_channel = torch.arange(h*w*d).reshape((1, 1, h, w, d)).to(self.device).float()
        self.index_channel /= self.index_channel.max()

        # Number of output channels in the first layer.
        c = 4

        # Output size, representing the maximum number of struts possible.
        self.shape_output = (
            16309,
        )
        size_output = int(np.prod(self.shape_output))

        self.convolution_1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*1, out_channels=c*2, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm3d(c*2),
            torch.nn.ReLU(inplace=True),
        )

        self.residual_1 = residual(c*2, c*2)
        self.convolution_bottleneck_1 = torch.nn.Conv3d(in_channels=c*2, out_channels=c*4, kernel_size=1)
        self.residual_2 = residual(c*4, c*4)
        self.convolution_bottleneck_2 = torch.nn.Conv3d(in_channels=c*4, out_channels=c*8, kernel_size=1)
        self.residual_3 = residual(c*8, c*8)
        self.convolution_bottleneck_3 = torch.nn.Conv3d(in_channels=c*8, out_channels=size_output, kernel_size=1)

        self.global_pooling = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([x, self.index_channel.expand((batch_size, -1, -1, -1, -1))], dim=1)

        x = self.convolution_1(x)
        x = self.convolution_2(x)

        x = torch.relu(x + self.residual_1(x))
        x = self.convolution_bottleneck_1(x)
        x = torch.relu(x + self.residual_2(x))
        x = self.convolution_bottleneck_2(x)
        x = torch.relu(x + self.residual_3(x))
        x = self.convolution_bottleneck_3(x)

        x = self.global_pooling(x)
        # x = torch.sigmoid(x)
        x = x.reshape((batch_size, *self.shape_output))

        return x

class ResNetLocal(torch.nn.Module):
    """3D ResNet-based CNN whose input is a 3D array of densities and whose output is a single strut diameter."""

    def __init__(self, device: str='cpu') -> None:
        super().__init__()
        self.device = device

        input_channels = 2

        # Number of output channels in the first layer.
        c = 4

        self.convolution_1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_4 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_5 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )

        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*1, c*1)
        self.residual_3 = residual(c*1, c*1)
        self.residual_4 = residual(c*1, c*1)
        self.residual_5 = residual(c*1, c*1)

        self.global_pooling = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.linear = torch.nn.Linear(in_features=c*1, out_features=1)

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

        x = self.global_pooling(x)[:, :, 0, 0, 0]
        x = self.linear(x)
        x = torch.sigmoid(x)

        return x

class Inception(torch.nn.Module):
    def __init__(self, device: str='cpu') -> None:
        super().__init__()

        input_channels = 2

        # Number of output channels in the first layer.
        c = 4

        self.convolution_1 = Sequential(
            Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=1, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_3 = Sequential(
            Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(),
        )
        self.convolution_5 = Sequential(
            Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=5, stride=1, padding='same'),
            BatchNorm3d(c*1),
            ReLU(),
        )
        # self.convolution_7 = Sequential(
        #     Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=7, stride=1, padding='same'),
        #     ReLU(),
        # )

        self.pooling = AvgPool3d((1, 1, 1))
        self.linear = Linear(in_features=(c*1)*3, out_features=1)

    def forward(self, x):
        x = torch.cat([
            self.convolution_1(x),
            self.convolution_3(x),
            self.convolution_5(x),
            # self.convolution_7(x),
        ], dim=1)
        x = self.pooling(x)[..., 0, 0, 0]
        x = self.linear(x)
        return x

# class Gnn(torch.nn.Module):
#     """GNN whose input is a graph of node densities and coordinates and whose output is a 2D tensor of strut diameters with shape (number of struts, 1)."""

#     def __init__(self, device: str='cpu') -> None:
#         super().__init__()

#         h = w = d = 11
#         input_channels = 1

#         # Model size, defined as the number of output channels in the first layer. Numbers of channels in subsequent layers are multiples of this number, so this number controls the overall model size.
#         c = 4

#         # Type of aggregation to use in the message passing layers.
#         aggregation = 'mean'

#         self.convolution = torch_geometric.nn.Sequential('x, edge_index', [
#             (torch_geometric.nn.GCNConv(in_channels=input_channels, out_channels=c*1, aggr=aggregation), 'x, edge_index -> x'),
#             torch.nn.ReLU(),
#             (torch_geometric.nn.GCNConv(in_channels=c*1, out_channels=c*2, aggr=aggregation), 'x, edge_index -> x'),
#             torch.nn.ReLU(),
#             (torch_geometric.nn.GCNConv(in_channels=c*2, out_channels=c*4, aggr=aggregation), 'x, edge_index -> x'),
#         ])

#         self.linear = torch.nn.Linear(in_features=c*4, out_features=1)
    
#     def forward(self, x, edge_index):
#         # Predict node embeddings with shape (total number of nodes across batch, number of node features).
#         x = self.convolution(x, edge_index)

#         # Average node features for each pair of nodes with resulting shape (number of edges, number of node features).
#         x = (x[edge_index[0, :], :] + x[edge_index[1, :], :])
#         # Combine node features for duplicate edges to reduce the shape to (half the original number of edges, ...).
#         x = x[:x.size(0)//2, :] + x[x.size(0)//2:, :]
#         # Predict values for each edge.
#         x = self.linear(x)

#         # # Predict edge values with shape (number of edges across batch,).
#         # x = torch.sum(
#         #     x,
#         #     dim=-1,
#         # )

#         # Average each pair of edges that represent the same strut (for example, (1, 2) and (2, 1)) with shape (half the original number of edges, 1). The second dimension is included for compatibility with the labels used during training.
#         # x = torch.mean(x.view([2, x.size(0)//2]), dim=0)[:, None]
#         # x = torch.mean(x.view([2, x.size(-1)//2]), dim=0)[:, None]

#         # Constrain output values to [0, 1].
#         # x = torch.sigmoid(x)

#         return x

# class Gnn0(torch.nn.Module):
#     """GNN without learnable parameters that does a simple averaging of adjacent nodes to calculate strut diameters.."""

#     def __init__(self, device: str=None) -> None:
#         super().__init__()

#         self.convolution = torch_geometric.nn.GCNConv(1, 1)

#     def forward(self, x, edge_index):
#         # Predict edge values with shape (number of edges across batch,).
#         x = torch.sum(
#             (x[edge_index[0, :], :] + x[edge_index[1, :], :]) / 2,
#             dim=-1,
#         )
#         # Average each pair of edges that represent the same strut (for example, (1, 2) and (2, 1)) with shape (half the original number of edges, 1). The second dimension is included for compatibility with the labels used during training.
#         x = torch.mean(x.view([2, x.size(-1)//2]), dim=0)[:, None]

#         return x