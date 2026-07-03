'''
This file defines the training loop for the GCN model. It includes:
- the monotonicity loss function to enforce the health index to be monotone with respect to the damage states
- the train function that sets up the model, loads the datasets, and runs the training loop with optional validation
- a plot_HI function to visualize the learned health index against the states after training

There is no cross validation implemented yet, only a single validation panel. The training and validation panels can be specified in the train function arguments.
You can train either with the big latent (the one used for reconstruction) or the sHI latent (the one we want to use for the GCN) by setting the big_latent argument in the train function. The default is False (train with sHI latent).
'''

import datetime
import os

from GCN import GraphCNN
from graph_dataset import Panel_GraphDataset, features_GraphDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from imagining_alghoritm import animate_panel, animate_panel_sidebyside

def monotonicity_loss(values, state_ids, panel_ids):
    # values:    (batch_size, k) — one or more scalars per graph (e.g. HI, or per-path outputs)
    # state_ids: (batch_size,)   — true damage-state index of each graph
    # panel_ids: (batch_size,)   — panel identity of each graph (state_ids restart at 0 per panel)
    #
    # Enforces a drop of 10 between *truly consecutive* states (same panel, state
    # index differs by exactly 1), regardless of the order samples arrived in the
    # batch (DataLoader shuffling means batch order is not state order).
    values = values.reshape(values.shape[0], -1)
    loss = values.sum() * 0.0
    for panel in panel_ids.unique():
        mask = panel_ids == panel
        if mask.sum() < 2:
            continue
        states = state_ids[mask]
        order = torch.argsort(states)
        sorted_states = states[order]
        sorted_values = values[mask][order]

        gaps = sorted_states[1:] - sorted_states[:-1]
        adjacent = gaps == 1
        if adjacent.sum() == 0:
            continue

        diff = sorted_values[1:] - sorted_values[:-1]
        diff = diff[adjacent]
        loss = loss + (((diff + 10.0) ** 2) - 100.0).sum()
    return loss



def combined_monotonicity_loss(HI, out, state_ids, panel_ids):
    # HI:  (batch_size, 1)          — graph-level health index
    # out: (batch_size * num_paths, 1) — per-node outputs, ordered by graph then node
    batch_size = HI.shape[0]
    num_paths = out.shape[0] // batch_size          # 28 for a 28-node graph
    path_out = out.reshape(batch_size, num_paths)   # (batch_size, 28)

    global_loss = monotonicity_loss(HI, state_ids, panel_ids)
    path_loss = monotonicity_loss(path_out, state_ids, panel_ids)
    return global_loss + path_loss

def combined_path_loss(HI, out, state_ids, panel_ids):
    # HI:  (batch_size, 1)          — graph-level health index
    # out: (batch_size * num_paths, 1) — per-node outputs, ordered by graph then node
    batch_size = HI.shape[0]
    num_paths = out.shape[0] // batch_size          # 28 for a 28-node graph
    path_out = out.reshape(batch_size, num_paths)   # (batch_size, 28)

    return monotonicity_loss(path_out, state_ids, panel_ids)

def plot_training_history(hist_dict, nr_epochs,save_dir=None):

    epochs = range(1, nr_epochs + 1)
    loss_history =  hist_dict['loss_history']
    val_loss_history = hist_dict['val_loss_history']
    global_loss_history = hist_dict['global_loss_history']
    global_val_loss_history = hist_dict['global_val_loss_history']
    path_loss_history = hist_dict['path_loss_history']
    path_val_loss_history = hist_dict['path_val_loss_history']

    # mark the lowest validation loss epoch
    lowest_val_epoch = torch.argmin(val_loss_history[:nr_epochs]).item() + 1

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(epochs, loss_history[:nr_epochs], label='Training Loss')
    ax[0].plot(epochs, val_loss_history[:nr_epochs], label='Validation Loss')
    ax[0].axvline(x=lowest_val_epoch, color='r', linestyle='--', label=f'Lowest Val Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Total Loss')
    ax[0].set_title('Total Loss')
    ax[0].legend()  

    ax[1].plot(epochs, global_loss_history[:nr_epochs], label='Training Global Loss')
    ax[1].plot(epochs, global_val_loss_history[:nr_epochs], label='Validation Global Loss')
    ax[1].axvline(x=lowest_val_epoch, color='r', linestyle='--', label=f'Lowest Val Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Global Loss')
    ax[1].set_title('Global Loss')
    ax[1].legend()

    ax[2].plot(epochs, path_loss_history[:nr_epochs], label='Training Path Loss')
    ax[2].plot(epochs, path_val_loss_history[:nr_epochs], label='Validation Path Loss')
    ax[2].axvline(x=lowest_val_epoch, color='r', linestyle='--', label=f'Lowest Val Loss')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Path Loss')
    ax[2].set_title('Path Loss')
    ax[2].legend()

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()

def train(f=3,val_panel=109,n_epochs=100,batch_size=15,lr=0.01,out_folder="GCN_models",big_latent=False, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
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

            global_loss = monotonicity_loss(HI, data.y, data.panel)
            path_loss = combined_path_loss(HI, out, data.y, data.panel)
            loss  = combined_monotonicity_loss(HI, out, data.y, data.panel)
            #loss = global_loss + path_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_global_loss += global_loss.item()
            total_path_loss += path_loss.item()
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

                global_val_loss = monotonicity_loss(HI_val, val_data.y, val_data.panel)
                path_val_loss = combined_path_loss(HI_val, out_val, val_data.y, val_data.panel)
                #loss_val = global_val_loss + path_val_loss
                loss_val = combined_monotonicity_loss(HI_val, out_val, val_data.y, val_data.panel)

                total_val_loss += loss_val.item()
                total_global_val_loss += global_val_loss.item()
                total_path_val_loss += path_val_loss.item()
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

def train_with_features(f=3,val_panel=109,n_epochs=100,batch_size=15,lr=0.01,out_folder="GCN_models", net_name="gcn_features.pt", cross_validation=False, seed=42):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
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

            global_loss = monotonicity_loss(HI, data.y, data.panel)
            path_loss = combined_path_loss(HI, out, data.y, data.panel)
            loss = global_loss + path_loss
            #loss = combined_monotonicity_loss(HI, out, data.y, data.panel)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_global_loss += global_loss.item()
            total_path_loss += path_loss.item()
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


                global_val_loss = monotonicity_loss(HI_val, val_data.y, val_data.panel)
                path_val_loss = combined_path_loss(HI_val, out_val, val_data.y, val_data.panel)
                loss_val = global_val_loss + path_val_loss
                #loss_val = combined_monotonicity_loss(HI_val, out_val, val_data.y, val_data.panel)


                total_val_loss += loss_val.item()
                total_global_val_loss += global_val_loss.item()
                total_path_val_loss += path_val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            avg_global_val_loss = total_global_val_loss / len(val_loader)
            avg_path_val_loss = total_path_val_loss / len(val_loader)
        
        #Save the model with the best validation loss
        if epoch == 0:
            best_val_loss = avg_val_loss
            torch.save({'model': model, 'feature_mean': feature_mean, 'feature_std': feature_std}, os.path.join(out_folder, net_name))
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({'model': model, 'feature_mean': feature_mean, 'feature_std': feature_std}, os.path.join(out_folder, net_name))
        
        
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

    if cross_validation:
        plot_output = os.path.join(out_folder, f"training_history_{net_name.split('.')[0]}_cv.png")
    plot_training_history(history_dict, epochs_done, save_dir=plot_output if cross_validation else None)


class EnsembleGCN(torch.nn.Module):
    """Wraps N cross-validation GCN models into one module.

    Each sub-model has its own normalization stats (feature_mean / feature_std).
    forward() normalizes x separately for each sub-model, averages the HI outputs,
    and also returns the per-model standard deviation as uncertainty estimate.
    Accepts raw (unnormalized) node features — do NOT apply a transform to the
    dataset when using this model.
    """
    def __init__(self, checkpoints):
        # checkpoints: list of {'model', 'feature_mean', 'feature_std'} dicts
        super().__init__()
        self.models = torch.nn.ModuleList([ck['model'] for ck in checkpoints])
        self.register_buffer('feature_means', torch.stack([ck['feature_mean'] for ck in checkpoints]))
        self.register_buffer('feature_stds',  torch.stack([ck['feature_std']  for ck in checkpoints]))

    def forward(self, x, edge_index, batch, edge_weight):
        all_out, all_HI = [], []
        for model, mean, std in zip(self.models, self.feature_means, self.feature_stds):
            x_norm = (x - mean) / std
            out, HI = model(x_norm, edge_index, batch, edge_weight)
            all_out.append(out)
            all_HI.append(HI)
        # out: (n_models, batch*n_paths, 1) → mean over models
        # HI:  (n_models, batch, 1)         → mean over models
        mean_out = torch.stack(all_out, dim=0).mean(dim=0)
        mean_HI  = torch.stack(all_HI,  dim=0).mean(dim=0)
        return mean_out, mean_HI


def build_and_save_ensemble(model_path, save_path):
    ensemble_filename = os.path.basename(save_path)
    checkpoints = [
        torch.load(os.path.join(model_path, f), weights_only=False)
        for f in sorted(os.listdir(model_path))
        if f.endswith('.pt') and f != ensemble_filename
    ]
    ensemble = EnsembleGCN(checkpoints)
    torch.save(ensemble, save_path)
    print(f"Ensemble saved to {save_path} ({len(checkpoints)} models)")
    return ensemble


def ensemble_predict(datasets, test_dataset, model_path, save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_datasets = datasets + [test_dataset]
    n = len(all_datasets)

    # HI predictions from each model: list of lists (dataset → model predictions)
    all_HI_per_dataset = [[] for _ in all_datasets]
    states_per_dataset = [None] * n

    for model_file in sorted(os.listdir(model_path)):
        if not model_file.endswith('.pt'):
            continue
        checkpoint = torch.load(os.path.join(model_path, model_file), weights_only=False)
        model = checkpoint['model'].to(device)
        feature_mean = checkpoint['feature_mean']
        feature_std = checkpoint['feature_std']
        model.eval()

        def norm_transform(data, mean=feature_mean, std=feature_std):
            data.x = (data.x - mean) / std
            return data

        for i, dataset in enumerate(all_datasets):
            dataset.transform = norm_transform
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
            batch_HI, batch_states = [], []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                    batch_HI.append(HI.cpu())
                    batch_states.append(data.y.cpu())
            all_HI_per_dataset[i].append(torch.cat(batch_HI).squeeze().numpy())
            if states_per_dataset[i] is None:
                states_per_dataset[i] = torch.cat(batch_states).squeeze().numpy()

    fig, ax = plt.subplots(1, n, figsize=(5 * n, 5))
    colors = ['steelblue'] * len(datasets) + ['orange']
    for i, dataset in enumerate(all_datasets):
        states = states_per_dataset[i]
        stacked = np.stack(all_HI_per_dataset[i], axis=0)  # (n_models, n_states)
        mean_HI = stacked.mean(axis=0)
        std_HI = stacked.std(axis=0)

        order = np.argsort(states)
        ax[i].fill_between(states[order], (mean_HI - std_HI)[order], (mean_HI + std_HI)[order],
                           alpha=0.25, color=colors[i])
        ax[i].scatter(states, mean_HI, alpha=0.7, s=15, color=colors[i])
        label = 'Test' if dataset is test_dataset else 'Train'
        ax[i].set_title(f'Panel {dataset.panel_number} ({label})')
        ax[i].grid(True)

    fig.suptitle('Ensemble Health Index vs Damage State')
    fig.supxlabel('Damage State Index')
    fig.supylabel('Health Index (HI)')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()

def plot_HI(model, train_datasets, validation_datasets, test_datasets, save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_len_datasets = len(train_datasets) + len(validation_datasets) + len(test_datasets)
    fig, ax = plt.subplots(1, total_len_datasets, figsize=(5 * total_len_datasets, 5))

    plot_idx = 0
    for dataset in train_datasets:
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

        ax[plot_idx].scatter(all_states, all_HI, alpha=0.7)
        ax[plot_idx].set_title(f'Panel {dataset.panel_number}')
        ax[plot_idx].grid(True)
        plot_idx += 1

    for dataset in validation_datasets:
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

        ax[plot_idx].scatter(all_states, all_HI, alpha=0.7, color='orange')
        ax[plot_idx].set_title(f'Panel {dataset.panel_number} (Validation)')
        ax[plot_idx].grid(True)
        plot_idx += 1

       
    for dataset in test_datasets:
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

        ax[plot_idx].scatter(all_states, all_HI, alpha=0.7, color='green')
        ax[plot_idx].set_title(f'Panel {dataset.panel_number} (Test)')
        ax[plot_idx].grid(True)
        plot_idx += 1
    fig.suptitle('Learned Health Index vs Damage State')
    fig.supxlabel('Damage State Index')
    fig.supylabel('Health Index (HI)')
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir)
    else:
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
    validation_panels = [103,104,105,109]

    if features: 
        dataset_103 = features_GraphDataset(root='graph_data', panel_number=103, freq=3)
        dataset_104 = features_GraphDataset(root='graph_data', panel_number=104, freq=3)
        dataset_105 = features_GraphDataset(root='graph_data', panel_number=105, freq=3)
        dataset_109 = features_GraphDataset(root='graph_data', panel_number=109, freq=3)
        dataset_123 = features_GraphDataset(root='graph_data', panel_number=123, freq=3)
    else:
        dataset_103 = Panel_GraphDataset(root='graph_data', panel_number=103, freq=3, big_latent=bl)
        dataset_104 = Panel_GraphDataset(root='graph_data', panel_number=104, freq=3, big_latent=bl)
        dataset_105 = Panel_GraphDataset(root='graph_data', panel_number=105, freq=3, big_latent=bl)
        dataset_109 = Panel_GraphDataset(root='graph_data', panel_number=109, freq=3, big_latent=bl)
        dataset_123 = Panel_GraphDataset(root='graph_data', panel_number=123, freq=3, big_latent=bl)

    dataset_dict = {
        103: dataset_103,
        104: dataset_104,
        105: dataset_105,
        109: dataset_109,
        123: dataset_123
    }  
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = f"Cross_Validation_Results/{current_date}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for val_panel in validation_panels:
        if features:
            model_name = "gcn_features_val_" + str(val_panel) + ".pt"
            train_with_features(batch_size=30, val_panel=val_panel, net_name=model_name, n_epochs=800, lr=0.001, out_folder=output_dir, cross_validation=True, seed=42)
            
        else:
            train(big_latent=bl, val_panel=val_panel, lr=0.0001, n_epochs=1000)
            if bl: 
                model_name = "gcn_big_latent.pt"
            else:
                model_name = "gcn.pt"
            

        bundle = torch.load(f"{output_dir}/{model_name}", weights_only=False)
       
        if isinstance(bundle, dict):
            model = bundle['model']
            feature_mean = bundle['feature_mean']
            feature_std  = bundle['feature_std']
            def norm_transform(data):
                data.x = (data.x - feature_mean) / feature_std
                return data
            for ds in dataset_dict.values():
                ds.transform = norm_transform
        else:
            model = bundle

        train_datasets = [dataset for key, dataset in dataset_dict.items() if key != val_panel and key != 123]
        validation_datasets = [dataset_dict[val_panel]]
        test_datasets = [dataset_dict[123]]  # Assuming panel 123 is always the test panel
        plot_HI(model, train_datasets, validation_datasets, test_datasets, save_dir=f"{output_dir}/HI_plot_val_{val_panel}.png")
    
    ensemble_predict([dataset_103, dataset_104, dataset_105, dataset_109], dataset_123, model_path=output_dir, save_dir=f"{output_dir}/ensemble_HI_plot.png")
    
    # Build and save ensemble model
    ensemble_model_path = os.path.join(output_dir, "ensemble_model.pt")
    build_and_save_ensemble(output_dir, ensemble_model_path)

    # Create 5 animation of panels using the ensemble model
    ensemble_model = torch.load(ensemble_model_path, weights_only=False)
    animate_panel_sidebyside(panel_number=103, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name=f"WCPDI_panel_103")
    animate_panel_sidebyside(panel_number=104, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name=f"WCPDI_panel_104")
    animate_panel_sidebyside(panel_number=105, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name=f"WCPDI_panel_105")
    animate_panel_sidebyside(panel_number=109, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name=f"WCPDI_panel_109")
    animate_panel_sidebyside(panel_number=123, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5,  features=features, transform=None, output_dir=output_dir, file_name=f"WCPDI_panel_123")
    
        # shi Plots for example paths
        # plot_sHI(model, dataset_109, path_idx=0)
        # plot_sHI(model, dataset_103, path_idx=0)
        # plot_sHI(model, dataset_123, path_idx=0)    