from typing import Tuple

import numpy as np
import torch
import torch_geometric


def get_parameter_count(model: torch.nn.Module) -> int:
    """Get the number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LatticeCnn(torch.nn.Module):
    """3D CNN whose input is a 3D array of densities and whose output is a 1D array of strut diameters."""

    def __init__(self, device: str='cpu') -> None:
        super().__init__()

        h = w = d = 11
        input_channels = 2
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
            int((h-strut_neighborhood_radius) * (w-strut_neighborhood_radius) * (d-strut_neighborhood_radius)),
            int((strut_neighborhood_radius+1) ** 3 - 1),
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
        x = torch.cat([x, torch.cat([self.index_channel] * batch_size, dim=0)], dim=1)

        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_bottleneck(x)

        x = self.global_pooling(x)
        x = torch.sigmoid(x)

        x = x.reshape((batch_size, *self.shape_output))

        return x

class LatticeResNet(torch.nn.Module):
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
            int((h-strut_neighborhood_radius) * (w-strut_neighborhood_radius) * (d-strut_neighborhood_radius)),
            int((strut_neighborhood_radius+1) ** 3 - 1),
        )
        size_output = int(np.prod(self.shape_output))

        # Function that returns a residual block.
        residual = lambda in_channels, out_channels: torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
        )

        self.convolution_1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=input_channels, out_channels=c*1, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm3d(c*1),
            torch.nn.ReLU(inplace=True),
        )
        self.convolution_2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=c*1, out_channels=c*2, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm3d(c*1),
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
        x = x.reshape((batch_size, *self.shape_output))

        return x

class LatticeGnn(torch.nn.Module):
    """A GNN whose input is a fully-connected graph of node locations and whose output is a graph of strut diameters."""

    def __init__(self, device: str='cpu') -> None:
        super().__init__()

        h = w = d = 11
        input_channels = 4

        # Model size, defined as the number of output channels in the first layer. Numbers of channels in subsequent layers are multiples of this number, so this number controls the overall model size.
        c = 1

        self.convolution = torch_geometric.nn.Sequential('x, edge_index', [
            (torch_geometric.nn.GCNConv(in_channels=input_channels, out_channels=c*1), 'x, edge_index -> x'),
            torch.nn.ReLU(),
            (torch_geometric.nn.GCNConv(in_channels=c*1, out_channels=c*2), 'x, edge_index -> x'),
            torch.nn.ReLU(),
            (torch_geometric.nn.GCNConv(in_channels=c*2, out_channels=c*4), 'x, edge_index -> x'),
        ])
    
    def forward(self, x, edge_index):
        # Predict node embeddings with shape (total number of nodes across batch, number of node features).
        x = self.convolution(x, edge_index)

        # Predict edge values with shape (number of edges across batch,).
        x = torch.sum(
            x[edge_index[0, :], :] * x[edge_index[1, :], :],
            dim=-1,
        )
        # Average each pair of edges that represent the same strut (for example, (1, 2) and (2, 1)) with shape (half the original number of edges, 1). The second dimension is included for compatibility with the labels used during training.
        x = torch.mean(x.view([2, x.size(-1)//2]), dim=0)[:, None]
        # x = x[:x.size(-1)//2][:, None]

        # Constrain output values to [0, 1].
        x = torch.sigmoid(x)

        return x



# """Test GNN architectures."""
# import torch
# import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
# from torch_geometric.nn import GCNConv


# dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)
# dataloader = DataLoader(dataset[:3], batch_size=1, shuffle=True)  # Must use PyG's DataLoader for compatibility with graphs

# print(f"Dataset of {len(dataset)} samples has {dataset.num_classes} classes, {dataset.num_node_features} node features, {dataset.num_node_attributes} node attributes, {dataset.num_edge_features} edge features, {dataset.num_edge_attributes} edge attributes.")


# class GNN(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         # # Convolve input image and produce output w/ same shape as input, then convert to graph; idea is to capture neighboring voxels' info into each voxel
#         # self.conv3d = torch.nn.Sequential(
#         #     torch.nn.Conv3d(1, 4, kernel_size=3, stride=1, padding='same'),
#         #     torch.nn.ReLU(inplace=True),
#         #     torch.nn.Conv3d(4, 8, kernel_size=3, stride=1, padding='same'),
#         # )

#         self.conv_1 = GCNConv(dataset.num_node_features, 32)
#         self.conv_2 = GCNConv(32, 8)
    
#     def forward(self, graph):
#         # input_image = self.conv3d(input_image)

#         x, edge_index = graph.x, graph.edge_index

#         # Predict node embeddings.
#         x = self.conv_1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv_2(x, edge_index)

#         # Predict edge values.
#         print(x.size())
#         x = (x[edge_index[0, :], :] * x[edge_index[1, :], :]).sum(dim=1)
#         print(x.size())

#         # Constrain output values to [0, 1].
#         x = torch.sigmoid(x)
#         print(x)

#         return x

# model = GNN()
# for batch in dataloader:
#     model(batch)
# # print(sum(len(p) for p in model.parameters()))
# # print(model.conv_1._parameters)