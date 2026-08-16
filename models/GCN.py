
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class DeepGraphCNN(torch.nn.Module):
    
    def __init__(self, num_node_features, hidden_channels=(32, 16), dropout=0.0,
                 use_residual=True, input_compress_dim=None):
        super().__init__()
        self.num_node_features = num_node_features
        self.use_residual = use_residual
        self.dropout = dropout
        self.input_compress_dim = input_compress_dim

        if input_compress_dim is not None:
            self.compress = torch.nn.Linear(num_node_features, input_compress_dim)
            conv_in_dim = input_compress_dim
        else:
            self.compress = None
            conv_in_dim = num_node_features

        dims = [conv_in_dim] + list(hidden_channels)
        self.convs = torch.nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        # Residual connections need matching in/out dims per block -- project the
        # block's input when its width changes, otherwise pass it through unchanged.
        self.res_projs = torch.nn.ModuleList([
            torch.nn.Identity() if dims[i] == dims[i + 1] else torch.nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        self.out_conv = GCNConv(dims[-1], 1)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if self.compress is not None:
            x = self.compress(x)
            x = F.leaky_relu(x, negative_slope=0.1)

        for conv, res_proj in zip(self.convs, self.res_projs):
            h = conv(x, edge_index, edge_weight)
            h = F.leaky_relu(h, negative_slope=0.1)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            x = res_proj(x) + h if self.use_residual else h

        x = self.out_conv(x, edge_index, edge_weight)
        HI = global_mean_pool(x, batch)
        return x, HI