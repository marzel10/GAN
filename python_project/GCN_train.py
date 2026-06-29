'''
This file defines the training loop for the GCN model. It includes:
- the monotonicity loss function to enforce the health index to be monotone with respect to the damage states
- the train function that sets up the model, loads the datasets, and runs the training loop with optional validation
- a plot_HI function to visualize the learned health index against the states after training

There is no cross validation implemented yet, only a single validation panel. The training and validation panels can be specified in the train function arguments.
You can train either with the big latent (the one used for reconstruction) or the sHI latent (the one we want to use for the GCN) by setting the big_latent argument in the train function. The default is False (train with sHI latent).
'''

import os

from GCN import GraphCNN
from graph_dataset import Panel_GraphDataset, features_GraphDataset
import torch
import matplotlib.pyplot as plt
import torch_geometric

def monotonicity_loss(HI, _state_indices):
    y_flat = HI.reshape(-1)
    diff = y_flat[1:] - y_flat[:-1]
    diff = diff + 10.0
    diff = diff ** 2
    baseline = 10 ** 2 * diff.shape[0]
    return diff.sum() - baseline



def combined_monotonicity_loss(HI, out):
    # HI:  (batch_size, 1)          — graph-level health index
    # out: (batch_size * num_paths, 1) — per-node outputs, ordered by graph then node
    batch_size = HI.shape[0]
    num_paths = out.shape[0] // batch_size          # 28 for a 28-node graph
    path_out = out.reshape(batch_size, num_paths)   # (batch_size, 28)

    global_loss = monotonicity_loss(HI, None)
    path_loss = sum(monotonicity_loss(path_out[:, i], None) for i in range(num_paths))
    return global_loss + path_loss

def combined_path_loss(HI, out):
    # HI:  (batch_size, 1)          — graph-level health index
    # out: (batch_size * num_paths, 1) — per-node outputs, ordered by graph then node
    batch_size = HI.shape[0]
    num_paths = out.shape[0] // batch_size          # 28 for a 28-node graph
    path_out = out.reshape(batch_size, num_paths)   # (batch_size, 28)

    
    path_loss = sum(monotonicity_loss(path_out[:, i], None) for i in range(num_paths))
    return path_loss

def plot_training_history(hist_dict, nr_epochs):

    epochs = range(1, nr_epochs + 1)
    loss_history = hist_dict['loss_history']
    val_loss_history = hist_dict['val_loss_history']
    global_loss_history = hist_dict['global_loss_history']
    global_val_loss_history = hist_dict['global_val_loss_history']
    path_loss_history = hist_dict['path_loss_history']
    path_val_loss_history = hist_dict['path_val_loss_history']


    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(epochs, loss_history[:nr_epochs], label='Training Loss')
    ax[0].plot(epochs, val_loss_history[:nr_epochs], label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Total Loss')
    ax[0].set_title('Total Loss')
    ax[0].legend()  

    ax[1].plot(epochs, global_loss_history[:nr_epochs], label='Training Global Loss')
    ax[1].plot(epochs, global_val_loss_history[:nr_epochs], label='Validation Global Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Global Loss')
    ax[1].set_title('Global Loss')
    ax[1].legend()

    ax[2].plot(epochs, path_loss_history[:nr_epochs], label='Training Path Loss')
    ax[2].plot(epochs, path_val_loss_history[:nr_epochs], label='Validation Path Loss')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Path Loss')
    ax[2].set_title('Path Loss')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

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
    loss_history = torch.zeros(n_epochs)
    global_loss_history = torch.zeros(n_epochs)
    path_loss_history = torch.zeros(n_epochs)
    val_loss_history = torch.zeros(n_epochs)
    global_val_loss_history = torch.zeros(n_epochs)
    path_val_loss_history = torch.zeros(n_epochs)

    epochs_done = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_global_loss = 0
        total_path_loss = 0
        for data in loader:
            data = data.to(device)

            optimizer.zero_grad()
            out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)

            global_loss = monotonicity_loss(HI, None)
            path_loss = combined_path_loss(HI, out)
            loss  = combined_monotonicity_loss(HI, out)
            #loss = global_loss + path_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_global_loss += global_loss
            total_path_loss += path_loss
        avg_loss = total_loss / len(loader)
        avg_global_loss = total_global_loss / len(loader)
        avg_path_loss = total_path_loss / len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_global_val_loss = 0
            total_path_val_loss = 0
            for val_data in val_loader:
                val_data = val_data.to(device)
                out_val, HI_val = model(val_data.x.to(device), val_data.edge_index.to(device), val_data.batch.to(device), val_data.edge_weight.to(device))

                global_val_loss = monotonicity_loss(HI_val, None)
                path_val_loss = combined_path_loss(HI_val, out_val)
                #loss_val = global_val_loss + path_val_loss
                loss_val = combined_monotonicity_loss(HI_val, out_val)

                total_val_loss += loss_val.item()
                total_global_val_loss += global_val_loss
                total_path_val_loss += path_val_loss
            avg_val_loss = total_val_loss / len(val_loader)
            avg_global_val_loss = total_global_val_loss / len(val_loader)
            avg_path_val_loss = total_path_val_loss / len(val_loader)

        loss_history[epoch] = avg_loss
        global_loss_history[epoch] = avg_global_loss
        path_loss_history[epoch] = avg_path_loss
        global_val_loss_history[epoch] = avg_global_val_loss
        path_val_loss_history[epoch] = avg_path_val_loss
        val_loss_history[epoch] = avg_val_loss
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        epochs_done += 1

    history_dict = {
        'loss_history': loss_history,
        'global_loss_history': global_loss_history,
        'path_loss_history': path_loss_history,
        'val_loss_history': val_loss_history,
        'global_val_loss_history': global_val_loss_history,
        'path_val_loss_history': path_val_loss_history}

    plot_training_history(history_dict, epochs_done)
   

    out_folder = out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if big_latent:
        torch.save(model, os.path.join(out_folder, "gcn_big_latent.pt"))
    else:
        torch.save(model, os.path.join(out_folder, "gcn.pt"))

def train_with_features(f=3,val_panel=109,n_epochs=100,batch_size=15,lr=0.01,out_folder="GCN_models", net_name="gcn_features.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features = 33
    model = GraphCNN(num_node_features=num_node_features).to(device)

    dataset_103 = features_GraphDataset(root='graph_data', panel_number=103, freq=f)
    dataset_104 = features_GraphDataset(root='graph_data', panel_number=104, freq=f)
    dataset_105 = features_GraphDataset(root='graph_data', panel_number=105, freq=f)
    dataset_109 = features_GraphDataset(root='graph_data', panel_number=109, freq=f)

    dict_datasets = {"103": dataset_103, "104": dataset_104, "105": dataset_105, "109": dataset_109}
    val_dataset = dict_datasets[str(val_panel)]

    # compute normalisation stats from training panels only
    all_train_datasets = [dataset for key, dataset in dict_datasets.items() if key != str(val_panel)]
    all_features = torch.cat([data.x for dataset in all_train_datasets for data in dataset], dim=0)
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0)
    feature_std[feature_std < 1e-10] = 1.0  # avoid division by zero for constant features

    # apply as a transform so it runs on every __getitem__ call (works with InMemoryDataset)
    def norm_transform(data):
        data.x = (data.x - feature_mean) / feature_std
        return data

    for dataset in dict_datasets.values():
        dataset.transform = norm_transform

    train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
    loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = torch.zeros(n_epochs)
    global_loss_history = torch.zeros(n_epochs)
    path_loss_history = torch.zeros(n_epochs)
    val_loss_history = torch.zeros(n_epochs)
    global_val_loss_history = torch.zeros(n_epochs)
    path_val_loss_history = torch.zeros(n_epochs)

    epochs_done = 0


    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_global_loss = 0
        total_path_loss = 0
        for data in loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)

            global_loss = monotonicity_loss(HI, None)
            path_loss = combined_path_loss(HI, out)
            #loss = global_loss + path_loss
            loss = combined_monotonicity_loss(HI, out)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_global_loss += global_loss
            total_path_loss += path_loss
        avg_loss = total_loss / len(loader)
        avg_global_loss = total_global_loss / len(loader)
        avg_path_loss = total_path_loss / len(loader)

        # Validation 
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_global_val_loss = 0
            total_path_val_loss = 0
            for val_data in val_loader:
                val_data = val_data.to(device)
                out_val, HI_val = model(val_data.x.to(device), val_data.edge_index.to(device), val_data.batch.to(device), val_data.edge_weight.to(device))
                

                global_val_loss = monotonicity_loss(HI_val, None)
                path_val_loss = combined_path_loss(HI_val, out_val)
                #loss_val = global_val_loss + path_val_loss
                loss_val = combined_monotonicity_loss(HI_val, out_val)


                total_val_loss += loss_val.item()
                total_global_val_loss += global_val_loss
                total_path_val_loss += path_val_loss
            avg_val_loss = total_val_loss / len(val_loader)
            avg_global_val_loss = total_global_val_loss / len(val_loader)
            avg_path_val_loss = total_path_val_loss / len(val_loader)
        
        loss_history[epoch] = avg_loss
        val_loss_history[epoch] = avg_val_loss
        global_loss_history[epoch] = avg_global_loss
        path_loss_history[epoch] = avg_path_loss
        global_val_loss_history[epoch] = avg_global_val_loss
        path_val_loss_history[epoch] = avg_path_val_loss

        print(f"Epoch: {epoch}, Loss: {avg_loss}, Val Loss: {avg_val_loss}")
        epochs_done += 1

    history_dict = {
        'loss_history': loss_history,
        'global_loss_history': global_loss_history,
        'path_loss_history': path_loss_history,
        'val_loss_history': val_loss_history,
        'global_val_loss_history': global_val_loss_history,
        'path_val_loss_history': path_val_loss_history}

    plot_training_history(history_dict, epochs_done)

    out_folder = out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    torch.save({'model': model, 'feature_mean': feature_mean, 'feature_std': feature_std}, os.path.join(out_folder, net_name))


def plot_HI(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
    all_HI = []
    all_states = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            all_HI.append(HI.cpu())
            all_states.append(data.y.cpu())
    all_HI = torch.cat(all_HI).squeeze().numpy()
    all_states = torch.cat(all_states).squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.scatter(all_states, all_HI, alpha=0.7)
    plt.xlabel('Damage State Index')
    plt.ylabel('Health Index (HI)')
    plt.title(f'Learned Health Index vs Damage State Panel {dataset.panel_number}')
    plt.grid(True)
    plt.show()

def plot_sHI(model, dataset, path_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
    all_sHI = []
    all_states = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, data.edge_weight)
            # out shape: (batch_size * num_paths, 1)
            num_paths = 28
            path_out = out.reshape(-1, num_paths)  # shape: (batch_size, num_paths)
            sHI_for_path = path_out[:, path_idx]    # shape: (batch_size,)
            all_sHI.append(sHI_for_path.cpu())
            all_states.append(data.y.cpu())
    all_sHI = torch.cat(all_sHI).squeeze().numpy()
    all_states = torch.cat(all_states).squeeze().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(all_states, all_sHI, alpha=0.7)
    plt.xlabel('Damage State Index')
    plt.ylabel(f'sHI for Path {path_idx + 1}')
    plt.title(f'Learned sHI vs Damage State Panel {dataset.panel_number} Path {path_idx + 1}')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    features = True
    bl =True
    if features:
        model_name = "gcn_features_lin.pt"
        train_with_features(batch_size=30,net_name=model_name, n_epochs=400, lr=0.001)
        
        dataset_123 = features_GraphDataset(root='graph_data', panel_number=123, freq=3)
        dataset_109 = features_GraphDataset(root='graph_data', panel_number=109, freq=3)
        dataset_103 = features_GraphDataset(root='graph_data', panel_number=103, freq=3)

    else:
        train(big_latent=bl, lr=0.0001, n_epochs=1000)
        if bl: 
            model_name = "gcn_big_latent.pt"
        else:
            model_name = "gcn.pt"
        
        dataset_109 = Panel_GraphDataset(root='graph_data', panel_number=109, freq=3, big_latent=bl)
        dataset_103 = Panel_GraphDataset(root='graph_data', panel_number=103, freq=3, big_latent=bl)
        dataset_123 = Panel_GraphDataset(root='graph_data', panel_number=123, freq=3, big_latent=bl)

    bundle = torch.load(f"GCN_models/{model_name}", weights_only=False)
    print(f"Bundle keys: {bundle.keys() if isinstance(bundle, dict) else 'Not a dict'}")
    if isinstance(bundle, dict):
        model = bundle['model']
        feature_mean = bundle['feature_mean']
        feature_std  = bundle['feature_std']
        def norm_transform(data):
            data.x = (data.x - feature_mean) / feature_std
            return data
        for ds in [dataset_109, dataset_103, dataset_123]:
            ds.transform = norm_transform
    else:
        model = bundle

    plot_HI(model, dataset_109)
    plot_HI(model, dataset_103)
    plot_HI(model, dataset_123)

    plot_sHI(model, dataset_109, path_idx=0)
    plot_sHI(model, dataset_103, path_idx=0)
    plot_sHI(model, dataset_123, path_idx=0)    