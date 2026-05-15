import os

from GCN import GraphCNN
from graph_dataset import Panel_GraphDataset
import torch
import matplotlib.pyplot as plt
import torch_geometric


def monotonicity_loss(HI, state_indices):
    """
    Penalises violations of monotonic increase in HI across damage states.

    HI:            (batch_size, 1) — one health index per graph in the batch.
    state_indices: (batch_size,)   — integer damage-state index for each graph.

    Strategy: sort HI by state index, then penalise any decrease between
    consecutive states.  Only violations (negative differences) contribute to
    the loss, so the network is free to be monotone without penalty.
    """
    hi = HI.squeeze(1)                          # (batch_size,)
    order = torch.argsort(state_indices)        # sort ascending by state
    hi_sorted = hi[order]
    diffs = hi_sorted[1:] - hi_sorted[:-1]     # negative = good (decrease), positive = bad (increase)
    violations = torch.relu(diffs)             # only penalise decreases
    return violations.mean()


def train(f=3,val_panel=109,n_epochs=100,batch_size=15,lr=0.01,out_folder="GCN_models",big_latent=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if big_latent:
        num_node_features = 15
    else:
        num_node_features = 1
    model = GraphCNN(num_node_features=num_node_features).to(device)

    dataset_103 = Panel_GraphDataset(root='graph_data', panel_number=103, freq=f, big_latent=big_latent)
    dataset_104 = Panel_GraphDataset(root='graph_data', panel_number=104, freq=f, big_latent=big_latent)
    dataset_105 = Panel_GraphDataset(root='graph_data', panel_number=105, freq=f, big_latent=big_latent)

    dataset_109 = Panel_GraphDataset(root='graph_data', panel_number=109, freq=f, big_latent=big_latent)
    dict_datasets = {"103": dataset_103, "104": dataset_104, "105": dataset_105, "109": dataset_109}
    val_dataset = dict_datasets[str(val_panel)]

    dataset = [dataset for key, dataset in dict_datasets.items() if key != str(val_panel)]
    dataset = torch.utils.data.ConcatDataset(dataset)
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    val_loss_history = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            out, HI = model(data.x, data.edge_index, data.batch)
            loss = monotonicity_loss(HI, data.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

        # Validation (optional)
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_data in val_loader:
                val_data = val_data.to(device)
                out_val, HI_val = model(val_data.x.to(device), val_data.edge_index.to(device), val_data.batch.to(device))
                loss_val = monotonicity_loss(HI_val, val_data.y.squeeze())
                
                total_val_loss += loss_val.item()
            avg_val_loss = total_val_loss / len(val_loader)

        
        loss_history.append(avg_loss)
        val_loss_history.append(avg_val_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Monotonicity Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    out_folder = out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if big_latent:
        torch.save(model, os.path.join(out_folder, "gcn_big_latent.pt"))
    else:
        torch.save(model, os.path.join(out_folder, "gcn.pt"))

def plot_HI(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
    all_HI = []
    all_states = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, HI = model(data.x, data.edge_index, data.batch)
            all_HI.append(HI.cpu())
            all_states.append(data.y.cpu())
    all_HI = torch.cat(all_HI).squeeze().numpy()
    all_states = torch.cat(all_states).squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.scatter(all_states, all_HI, alpha=0.7)
    plt.xlabel('Damage State Index')
    plt.ylabel('Health Index (HI)')
    plt.title('Learned Health Index vs Damage State')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bl =True
    train(big_latent=bl, lr=0.0001, n_epochs=500)
    if bl: 
        model_name = "gcn_big_latent.pt"
    else:
        model_name = "gcn.pt"
    model = torch.load(f"GCN_models/{model_name}", weights_only=False)
    dataset_109 = Panel_GraphDataset(root='graph_data', panel_number=109, freq=3, big_latent=bl)
    plot_HI(model, dataset_109)