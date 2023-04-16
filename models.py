from typing import *

import numpy as np
import torch
from torch.nn import *
# import torch_geometric


def print_model_summary(model: Module) -> None:
    """Print information about a model."""
    print(f"\n{type(model).__name__}")
    print(f"\tTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\tLearnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Function that returns a residual block.
def residual(in_channels, out_channels):
    return Sequential(
        Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
        BatchNorm3d(num_features=out_channels),
        ReLU(inplace=False),
        Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
    )

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
        # x = torch.sigmoid(x)

        return x

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

class UNet(Module):
    def __init__(self) -> None:
        super().__init__()

        c = 4

        self.convolution_1 = Sequential(
            Conv3d(2, c*1, 3, 1, 1),
            BatchNorm3d(c*1),
            ReLU(),
            Conv3d(c*1, c*1, 3, 1, 1),
            BatchNorm3d(c*1),
            ReLU(),
        )
        self.convolution_2 = Sequential(
            Conv3d(c*1, c*2, 3, 2, 1),
            BatchNorm3d(c*2),
            ReLU(),
            Conv3d(c*2, c*2, 3, 1, 1),
            BatchNorm3d(c*2),
            ReLU(),
        )
        self.convolution_3 = Sequential(
            Conv3d(c*2, c*4, 3, 2, 1),
            BatchNorm3d(c*4),
            ReLU(),
            Conv3d(c*4, c*4, 3, 1, 1),
            BatchNorm3d(c*4),
            ReLU(),
        )
        self.deconvolution_1 = Sequential(
            ConvTranspose3d(c*4+c*2, c*4, 3, 1, 1),
            BatchNorm3d(c*4),
            ReLU(),
            ConvTranspose3d(c*4, c*2, 3, 2, 1, output_padding=1),
            BatchNorm3d(c*2),
            ReLU(),
        )
        self.deconvolution_2 = Sequential(
            ConvTranspose3d(c*2+c*1, c*2, 3, 1, 1),
            BatchNorm3d(c*2),
            ReLU(),
            ConvTranspose3d(c*2, c*1, 3, 2, 1),
            BatchNorm3d(c*1),
            ReLU(),
        )
        self.deconvolution_3 = Sequential(
            ConvTranspose3d(c*1, c*1, 3, 1, 1),
            BatchNorm3d(c*1),
            ReLU(),
            ConvTranspose3d(c*1, 13, 3, 1, 1),
        )

    def forward(self, x):
        x_1 = self.convolution_1(x)
        x_2 = self.convolution_2(x_1)
        x_3 = self.convolution_3(x_2)
        x_4 = self.deconvolution_1(torch.cat([x_2[..., 1:4, 1:4, 1:4], x_3], dim=1))
        x_5 = self.deconvolution_2(torch.cat([x_1[..., 3:9, 3:9, 3:9], x_4], dim=1))
        x_6 = self.deconvolution_3(x_5)
        x = torch.clip(x_6, 0, 100)
        return x

class NoRelu(Module):
    def __init__(self) -> None:
        super().__init__()

        c = 4

        self.layers = Sequential(
            Conv3d(2, c*1, 3, 1, 1),
            BatchNorm3d(c*1),
            Conv3d(c*1, c*2, 3, 1, 1),
            BatchNorm3d(c*2),
            Conv3d(c*2, 13, 3, 1, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.clip(x, 0, 100)
        return x

class TwoBranch(Module):
    def __init__(self) -> None:
        super().__init__()

        # self.branch_1 = Sequential(
        #     Conv3d(1, 4, 3, 1, 'same', padding_mode='reflect'),
        #     BatchNorm3d(4),
        #     ReLU(inplace=True),
        #     Conv3d(4, 8, 3, 1, 'same', padding_mode='zeros'),
        #     BatchNorm3d(8),
        #     ReLU(inplace=True),
        #     Conv3d(8, 13, 3, 1, 'same', padding_mode='zeros'),
        #     BatchNorm3d(13),
        #     ReLU(inplace=True),
        # )

        self.branch_2 = Sequential(
            Conv3d(1, 1, 3, 1, 'same'),
        )

        self.convolution_1 = Sequential(
            Conv3d(1, 4, 3, 1, 'same'),
            BatchNorm3d(4),
            ReLU(inplace=True),
        )
        self.residual_1 = residual(4, 4)
        self.convolution_2 = Sequential(
            Conv3d(4, 8, 1, 1, 'same'),
            BatchNorm3d(8),
            ReLU(inplace=True),
        )
        self.residual_2 = residual(8, 8)
        self.convolution_3 = Sequential(
            Conv3d(8, 13, 1, 1, 'same'),
            BatchNorm3d(13),
            ReLU(inplace=True),
        )
        self.residual_3 = residual(13, 13)
    
    def forward(self, x):
        x1 = self.convolution_1(x[:, 0:1, ...])
        x1 = torch.relu(x1 + self.residual_1(x1))
        x1 = self.convolution_2(x1)
        x1 = torch.relu(x1 + self.residual_2(x1))
        x1 = self.convolution_3(x1)
        x1 = torch.relu(x1 + self.residual_3(x1))

        x2 = self.branch_2(x[:, 1:2, ...])

        x = x1 * x2
        x = torch.clip(x, 0, 100)
        return x

class ThirteenBranch(Module):
    def __init__(self) -> None:
        super().__init__()

        self.branch_1 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_2 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_3 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_4 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_5 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_6 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_7 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_8 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_9 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_10 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_11 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_12 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
        self.branch_13 = Sequential(Conv3d(2, 1, 5, 1, padding='same'))
    
    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        x5 = self.branch_5(x)
        x6 = self.branch_6(x)
        x7 = self.branch_7(x)
        x8 = self.branch_8(x)
        x9 = self.branch_9(x)
        x10 = self.branch_10(x)
        x11 = self.branch_11(x)
        x12 = self.branch_12(x)
        x13 = self.branch_13(x)
        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13], dim=1)
        x = torch.clip(x, 0, 100)
        return x

# class ResNetMasked(Module):
#     """Variant of ResNet that only sends two locations in the input tensor into the fully-connected layer."""

#     def __init__(self) -> None:
#         super().__init__()

#         input_channels = 2

#         # Number of output channels in the first layer.
#         c = 4

#         self.convolution_1 = Sequential(
#             Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
#             BatchNorm3d(c*1),
#             ReLU(inplace=True),
#         )
#         self.convolution_2 = Sequential(
#             Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
#             BatchNorm3d(c*1),
#             ReLU(inplace=True),
#         )
#         self.convolution_3 = Sequential(
#             Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
#             BatchNorm3d(c*1),
#             ReLU(inplace=True),
#         )
#         self.convolution_4 = Sequential(
#             Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
#             BatchNorm3d(c*1),
#             ReLU(inplace=True),
#         )
#         self.convolution_5 = Sequential(
#             Conv3d(in_channels=c*1, out_channels=c*1, kernel_size=3, stride=1, padding='same'),
#             BatchNorm3d(c*1),
#             ReLU(inplace=True),
#         )

#         self.residual_1 = residual(c*1, c*1)
#         self.residual_2 = residual(c*1, c*1)
#         self.residual_3 = residual(c*1, c*1)
#         self.residual_4 = residual(c*1, c*1)
#         self.residual_5 = residual(c*1, c*1)

#         self.linear = Linear(in_features=c*1, out_features=1)

#     def forward(self, x, coordinates):
#         batch_size = x.size(0)

#         x = self.convolution_1(x)
#         x = torch.relu(x + self.residual_1(x))
#         x = self.convolution_2(x)
#         x = torch.relu(x + self.residual_2(x))
#         x = self.convolution_3(x)
#         x = torch.relu(x + self.residual_3(x))
#         x = self.convolution_4(x)
#         x = torch.relu(x + self.residual_4(x))
#         x = self.convolution_5(x)
#         x = torch.relu(x + self.residual_5(x))

#         (x1, y1, z1), (x2, y2, z2) = coordinates
#         batch_indices = torch.arange(batch_size)
#         # Take the average of two voxels in each channel.
#         x = (x[batch_indices, :, x1, y1, z1] + x[batch_indices, :, x2, y2, z2]) / 2
#         # x = torch.mean([
#         #     x[batch_indices, :, x1, y1, z1],
#         #     x[batch_indices, :, x2, y2, z2],
#         # ], dim=0)
#         x = self.linear(x.view(batch_size, -1))

#         return x

# class MLP(Module):
#     def __init__(self) -> None:
#         super().__init__()

#         self.linear = Sequential(
#             Linear(11**3, 16),
#             ReLU(inplace=True),
#             Linear(16, 8),
#             ReLU(inplace=True),
#             Linear(8, 1),
#         )

#         # Initialize weights that linearly decrease away from the center of the volume.
#         coordinates = np.indices((11, 11, 11)) - 5
#         coordinates = np.sqrt(np.sum((coordinates ** 2), axis=0))
#         coordinates = 1 - (coordinates / coordinates.max())
#         coordinates *= 0.1
#         with torch.no_grad():
#             self.linear.get_parameter('0.weight')[:] = torch.tensor(coordinates[..., 1].flatten()[None, 1], requires_grad=True)
    
#     def forward(self, x):
#         x = self.linear(x.view(x.size(0), -1))
#         return x

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