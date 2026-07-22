
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
class GraphCNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GraphCNN, self).__init__()
        self.num_node_features = num_node_features
        if num_node_features > 24:
            self.lin = torch.nn.Linear(num_node_features, 24)
            self.conv1 = GCNConv(24, 1)
        else:
            self.conv1 = GCNConv(num_node_features, 1)
        

    def forward(self, x, edge_index, batch, edge_weight=None):
        if self.num_node_features > 24:
            x = self.lin(x)
            x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv1(x, edge_index, edge_weight)
        


        
        HI = global_mean_pool(x, batch)  # Pooling to get graph-level representation
        #HI = self.lin(HI)  # Final output
        return x, HI


class DeepGraphCNN(torch.nn.Module):
    """Deeper variant of GraphCNN: N stacked GCNConv layers (depth/width set by
    hidden_channels) instead of a single conv straight to the scalar output, with
    optional residual connections to counter over-smoothing -- a real risk here since
    the graph is small and densely connected (28 nodes, crossing-based edges), so a
    few hops can already mix most of the graph together. Kept separate from GraphCNN
    so existing GCN_train.py callers (train / train_with_features) are unaffected.

    Same forward interface as GraphCNN: returns (x, HI) where x is the per-node scalar
    output (batch*num_paths, 1) and HI is its graph-level mean-pooled value -- both are
    still required downstream (combined_path_loss, damage_map_loss expect exactly one
    value per node).

    input_compress_dim, if given, restores GraphCNN's own behavior: a plain (non-graph-
    aware) Linear + LeakyReLU compresses raw node features to this width BEFORE any
    GCNConv layer runs, so the GCNConv stack always sees a fixed-width input regardless
    of num_node_features -- e.g. input_compress_dim=24 makes this a drop-in deeper
    replacement for GraphCNN's own Linear(num_node_features, 24) step. None (default)
    skips it and feeds num_node_features directly into the first GCNConv, as before.
    """
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