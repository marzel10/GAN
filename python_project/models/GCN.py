
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
class GraphCNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GraphCNN, self).__init__()
        self.num_node_features = num_node_features
        if num_node_features > 15:
            self.lin = torch.nn.Linear(num_node_features, 15)
            self.conv1 = GCNConv(15, 1)
        else:
            self.conv1 = GCNConv(num_node_features, 1)
        

    def forward(self, x, edge_index, batch, edge_weight=None):
        if self.num_node_features > 15:
            x = self.lin(x)
            x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv1(x, edge_index, edge_weight)
        


        
        HI = global_mean_pool(x, batch)  # Pooling to get graph-level representation
        #HI = self.lin(HI)  # Final output
        return x, HI