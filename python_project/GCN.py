
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
class GraphCNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GraphCNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 1)
        #self.lin = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        HI = global_mean_pool(x, batch)  # Pooling to get graph-level representation
        #HI = self.lin(HI)  # Final output
        return x, HI