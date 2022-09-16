from typing import Tuple

import numpy as np
import torch
import torch_geometric


def get_parameter_count(model: torch.nn.Module) -> int:
    """Get the number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LatticeCnn(torch.nn.Module):
    """3D CNN whose input is a 3D array of densities and whose output is a 1D array of strut diameters."""

    def __init__(self) -> None:
        super().__init__()

        h, w, d = 51, 51, 51
        input_channels = 2
        strut_neighborhood = (3, 3, 3)

        self.index_channel = torch.arange(h*w*d).reshape((1, 1, h, w, d)).to('cpu').float()
        self.index_channel /= self.index_channel.max() * 255

        # Number of output channels in the first layer.
        c = 2

        # Parameters for all convolution layers.
        k = 5
        s = 2
        p = 1

        # Output size, representing the maximum number of struts possible.
        self.shape_output = (
            np.prod([dimension - (strut_neighborhood[i] - 1) for i, dimension in enumerate((h, w, d))]),
            np.prod(strut_neighborhood) - 1,
        )
        size_output = np.prod(self.shape_output)

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

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=size_convolution, out_features=size_output),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([x, torch.cat([self.index_channel] * batch_size, dim=0)], dim=1)

        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.linear(x.view(batch_size, -1))
        x = x.reshape((batch_size, *self.shape_output))

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

class LatticeGnn(torch.nn.Module):
    """A GNN whose input is a fully-connected graph of node locations and whose output is a graph of strut diameters."""

    def __init__(self) -> None:
        super().__init__()

        h, w, d = 51, 51, 51
        input_channels = 1

        # Model size, defined as the number of output channels in the first layer. Numbers of channels in subsequent layers are multiples of this number, so this number controls the overall model size.
        c = 8

        self.convolution = torch_geometric.nn.Sequential('x, edge_index', [
            (torch_geometric.nn.GCNConv(in_channels=input_channels, out_channels=c*1), 'x, edge_index -> x'),
            torch.nn.ReLU(),
            (torch_geometric.nn.GCNConv(in_channels=c*1, out_channels=c*2), 'x, edge_index -> x'),
            torch.nn.ReLU(),
            (torch_geometric.nn.GCNConv(in_channels=c*2, out_channels=c*4), 'x, edge_index -> x'),
        ])
    
    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        # Predict node embeddings (number of nodes, number of node features).
        x = self.convolution(x, edge_index)

        # Predict edge values (number of edges,).
        x = torch.sum(x[edge_index[0, :], :] * x[edge_index[1, :], :], dim=-1)

        # Constrain output values to [0, 1].
        x = torch.sigmoid(x)

        return x



"""Test GNN architectures."""
import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv


dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)
dataloader = DataLoader(dataset[:3], batch_size=1, shuffle=True)  # Must use PyG's DataLoader for compatibility with graphs

print(f"Dataset of {len(dataset)} samples has {dataset.num_classes} classes, {dataset.num_node_features} node features, {dataset.num_node_attributes} node attributes, {dataset.num_edge_features} edge features, {dataset.num_edge_attributes} edge attributes.")


class GNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # # Convolve input image and produce output w/ same shape as input, then convert to graph; idea is to capture neighboring voxels' info into each voxel
        # self.conv3d = torch.nn.Sequential(
        #     torch.nn.Conv3d(1, 4, kernel_size=3, stride=1, padding='same'),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv3d(4, 8, kernel_size=3, stride=1, padding='same'),
        # )

        self.conv_1 = GCNConv(dataset.num_node_features, 32)
        self.conv_2 = GCNConv(32, 8)
    
    def forward(self, graph):
        # input_image = self.conv3d(input_image)

        x, edge_index = graph.x, graph.edge_index

        # Predict node embeddings.
        x = self.conv_1(x, edge_index)
        x = torch.relu(x)
        x = self.conv_2(x, edge_index)

        # Predict edge values.
        print(x.size())
        x = (x[edge_index[0, :], :] * x[edge_index[1, :], :]).sum(dim=1)
        print(x.size())

        # Constrain output values to [0, 1].
        x = torch.sigmoid(x)
        print(x)

        return x

model = GNN()
for batch in dataloader:
    model(batch)
# print(sum(len(p) for p in model.parameters()))
# print(model.conv_1._parameters)