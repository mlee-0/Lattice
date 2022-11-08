from typing import Tuple

import numpy as np
import torch
from torch.nn import *
# import torch_geometric


def get_parameter_count(model: Module) -> int:
    """Get the number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function that returns a residual block.
def residual(in_channels, out_channels):
    return Sequential(
        Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
        ReLU(inplace=False),
        Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
    )


class ResNet(Module):
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

        # # Set certain weights to 0.
        # with torch.no_grad():
        #     self.convolution_2.get_parameter('0.weight')[..., 1, 1, 1] = 0
        #     self.convolution_3.get_parameter('0.weight')[..., 1, 1, 1] = 0
        #     self.convolution_4.get_parameter('0.weight')[..., 1, 1, 1] = 0
        #     self.convolution_5.get_parameter('0.weight')[..., 1, 1, 1] = 0
        #     for layer in [self.residual_1, self.residual_2, self.residual_3, self.residual_4, self.residual_5]:
        #         layer.get_parameter('0.weight')[..., 1, 1, 1] = 0
        #         layer.get_parameter('2.weight')[..., 1, 1, 1] = 0

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
        # x = torch.sigmoid(x)

        return x

class ResNetMasked(Module):
    """Variant of ResNet that only sends two locations in the input tensor into the fully-connected layer."""

    def __init__(self) -> None:
        super().__init__()

        input_channels = 1

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
        # self.convolution_4 = Sequential(
        #     Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
        #     BatchNorm3d(c*1),
        #     ReLU(inplace=True),
        # )
        # self.convolution_5 = Sequential(
        #     Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
        #     BatchNorm3d(c*1),
        #     ReLU(inplace=True),
        # )

        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*1, c*1)
        self.residual_3 = residual(c*1, c*1)
        # self.residual_4 = residual(c*1, c*1)
        # self.residual_5 = residual(c*1, c*1)

        # self.pooling = AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.linear = Linear(in_features=c*1, out_features=1)

    def forward(self, x, coordinates):
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

        (x1, y1, z1), (x2, y2, z2) = coordinates
        batch_indices = torch.arange(batch_size)
        x = (x[batch_indices, :, x1, y1, z1] + x[batch_indices, :, x2, y2, z2]) / 2
        # x = torch.mean([
        #     x[batch_indices, :, x1, y1, z1],
        #     x[batch_indices, :, x2, y2, z2],
        # ], dim=0)
        # x = self.pooling(x)
        x = self.linear(x.view(batch_size, -1))

        return x

class MLP(Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = Sequential(
            Linear(11**3, 1),
            # Linear(256, 64),
            # Linear(64, 32),
            # Linear(32, 16),
            # Linear(16, 8),
            # Linear(8, 1),
        )

        # Initialize weights that linearly decrease away from the center of the volume.
        coordinates = np.indices((11, 11, 11)) - 5
        coordinates = np.sqrt(np.sum((coordinates ** 2), axis=0))
        coordinates = 1 - (coordinates / coordinates.max())
        coordinates *= 0.1
        with torch.no_grad():
            self.linear.get_parameter('0.weight')[:] = torch.tensor(coordinates[..., 1].flatten()[None, 1], requires_grad=True)
    
    def forward(self, x):
        x = self.linear(x.view(x.size(0), -1))
        return x

# class Gnn(torch.nn.Module):
#     """GNN whose input is a graph of node densities and coordinates and whose output is a 2D tensor of strut diameters with shape (number of struts, 1)."""

#     def __init__(self) -> None:
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

# class EdgeConv(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         input_channels = 1

#         self.convolution_1 = torch_geometric.nn.EdgeConv(
#             torch_geometric.nn.MLP([2*input_channels, 4]),
#             aggr='mean',
#         )
#         self.convolution_2 = torch_geometric.nn.EdgeConv(
#             torch_geometric.nn.MLP([4*2, 1]),
#             aggr='mean',
#         )
#         self.convolution_3 = torch_geometric.nn.EdgeConv(
#             torch_geometric.nn.MLP([16*2, 1]),
#             aggr='mean',
#         )

#     def forward(self, x, edge_index):
#         x = self.convolution_1(x, edge_index)
#         x = self.convolution_2(x, edge_index)
#         x = self.convolution_3(x, edge_index)

#         # Combine node features for each pair of nodes with resulting shape (number of edges, number of node features).
#         x = (x[edge_index[0, :], :] + x[edge_index[1, :], :]) / 2
#         # Combine node features for duplicate edges to reduce the shape to (half the original number of edges, ...).
#         x = (x[:x.size(0)//2, :] + x[x.size(0)//2:, :]) / 2

#         return x

# class CnnGnn(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, density, edge_index):
#         x = self.convolution_1(density)