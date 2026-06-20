'''
This file:
- defines the Panel_GraphDataset class for creating graph datasets from the extracted sHI values and attention-based adjacency matrices
- the dataset is created for each state, where nodes represent paths and edges represent crossing paths weighted by the attention values
- includes a main block that demonstrates how to load the dataset and print some example graph data for a given panel and frequency
- by default, it can load either the sHI features or the big latent features extracted from the autoencoder, depending on the big_latent flag

If you do any changes in the process method, make sure to delete the processed files to trigger re-processing with the new code.
'''

import warnings
# format error messages with panel and state information for easier debugging
warnings.formatwarning = lambda msg, *_, **__: f"Warning: {msg}\n"

from torch_geometric.data import Data, InMemoryDataset
import torch
from weight_matrix import find_crossings, adjencency_matrix
import numpy as np
from states_check import states, sampling_rate
from features_extractor import FeaturesExtractor
from pathlib import Path


NR_SAVED_STATES ={"123_1": 7, "123_2": 7, "123_31": 16, "123_32": 13, "123_41": 10, "123_42": 10, "123_43": 10, "123_44": 10}
STATE_START_INDICES = {"123_1": 0, "123_2": 7, "123_31": 14, "123_32": 30, "123_41": 43, "123_42": 53, "123_43": 63, "123_44": 73}
class Panel_GraphDataset(InMemoryDataset):
    def __init__(self, root, panel_number,freq, big_latent=False, transform=None, pre_transform=None):
        self.panel_number = panel_number
        self.freq = freq
        self.big_latent = big_latent
        super().__init__(root, transform, pre_transform)
        
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"panel_{self.panel_number}_shi_raw_{self.freq}.pt"]

    @property
    def processed_file_names(self):
        if self.big_latent:
            return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}.pt"]
        return [f"panel_{self.panel_number}_processed_{self.freq}.pt"]


    def process(self):
        # Load raw data (this should be implemented to read your specific raw format)
        raw_data = torch.load(self.raw_paths[0], weights_only=False)
        
        features = np.array(raw_data["shi"]) # shape: paths x states x 1
        path_labels = raw_data["path_labels"] # shape: paths
        big_latent = raw_data["big_latent"] # shape: paths x states x latent_dim

        # Process raw_data into a list of Data objects
        data_list = []
        for state in range(features.shape[1]):
            if self.big_latent:
                x = torch.tensor(big_latent[:, state, :], dtype=torch.float)  # shape: (num_paths, latent_dim)
            else:
                x = torch.tensor(features[:, state, 0], dtype=torch.float).unsqueeze(1)  # shape: (num_paths, 1)

            if (x == 0).all():
                warnings.warn(f"Warning: All features are zero for state {state}. Check if the raw features are correct.")
            
            connection_matrix = find_crossings()
            edge_index = torch.tensor(np.array(np.nonzero(connection_matrix)), dtype=torch.long)  # shape: 2 x num_edges


            _DATA_DIR = Path(__file__).resolve().parent

            if str(self.panel_number) == "123" :
                    # find which subpanel this state belongs to
                    subpanel = None
                    for sp in NR_SAVED_STATES.keys():
                        start_idx = STATE_START_INDICES[sp]
                        end_idx = start_idx + NR_SAVED_STATES[sp]
                        if start_idx <= state <= end_idx:
                            subpanel = sp
                            print(f"State {state} belongs to subpanel {subpanel}")
                            break
                    if subpanel is None:
                        raise ValueError(f"State index {state} does not belong to any known subpanel.")
                    current_state = states(str(_DATA_DIR / f"data/States_{subpanel}.mat"))
            else:
                current_state = states(str(_DATA_DIR / f"data/States_{self.panel_number}.mat"))
            
            adj_matrix = adjencency_matrix(current_state, state_idx=state, freq_idx=self.freq)  # shape: paths x paths

            if (adj_matrix == 0).all():
                warnings.warn(f"Warning: Adjacency matrix for state {state} is all zeros. Check if the attention values are correct.")
            
            
            adj_matrix = adj_matrix[connection_matrix==1]  # shape: num_edges
            edge_weight = torch.tensor((np.array(adj_matrix)), dtype=torch.float)
            
            # if you want to implement labels in the future 
            # y = labels  # shape: pathsxoutput_dim

            data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                                  y=torch.tensor([state], dtype=torch.long)))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class features_GraphDataset(InMemoryDataset):
    def __init__(self, root, panel_number,freq, transform=None, pre_transform=None):
        self.panel_number = panel_number
        self.freq = freq
        
        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.panel_number == 123:
            #return list of all subpanels
            return [f"features_cache/States_{sp}_freq{self.freq}_all_paths.npy" for sp in ["123_1", "123_2", "123_31", "123_32", "123_41", "123_42", "123_43", "123_44"]]
        else:
            return [f"features_cache/States_{self.panel_number}_freq{self.freq}_all_paths.npy"]
    
    @property
    def processed_file_names(self):
        return [f"features_cache/panel_{self.panel_number}_processed_features_{self.freq}.pt"]
    
    def process(self):
        if self.panel_number == 123:
            features_list = []
            for file in self.raw_file_names:
                features = np.load(file)
                features_list.append(features)
            features = np.concatenate(features_list, axis=1)  # shape: paths x states x features
            print(f"Concatenated features shape for panel 123: {features.shape}")
        else:
            features = np.load(self.raw_file_names[0])  # shape: paths x states x features
            print(f"Loaded features shape for panel {self.panel_number}: {features.shape}")

        # Process features into a list of Data objects
        data_list = []
        for state in range(features.shape[1]):
            x = torch.tensor(features[:, state, :], dtype=torch.float)  # shape: (num_paths, num_features)
            
            # Create edge weights
            connection_matrix = find_crossings()
            edge_index = torch.tensor(np.array(np.nonzero(connection_matrix)), dtype=torch.long)
            _DATA_DIR = Path(__file__).resolve().parent
            if str(self.panel_number) == "123" :
                    # find which subpanel this state belongs to
                    subpanel = None
                    for sp in NR_SAVED_STATES.keys():
                        start_idx = STATE_START_INDICES[sp]
                        end_idx = start_idx + NR_SAVED_STATES[sp]
                        if start_idx <= state <= end_idx:
                            subpanel = sp
                            print(f"State {state} belongs to subpanel {subpanel}")
                            break
                    if subpanel is None:
                        raise ValueError(f"State index {state} does not belong to any known subpanel.")
                    current_state = states(str(_DATA_DIR / f"data/States_{subpanel}.mat"))
            else:
                current_state = states(str(_DATA_DIR / f"data/States_{self.panel_number}.mat"))

            adj_matrix = adjencency_matrix(current_state, state_idx=state, freq_idx=self.freq)  # shape: paths x paths
            adj_matrix = adj_matrix[connection_matrix==1]  # shape: num_edges
            edge_weight = torch.tensor((np.array(adj_matrix)), dtype=torch.float)
            data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                                  y=torch.tensor([state], dtype=torch.long)))
        data, slices = self.collate(data_list)

        # make sure the processed directory exists
        processed_dir = Path(self.processed_paths[0]).parent
        processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])        


if __name__ == "__main__":
    '''
    dataset_109 = Panel_GraphDataset(root='graph_data', panel_number=109, freq=3, big_latent=True)
    print(f"Dataset loaded with {len(dataset_109)} states.")
    print(f"Example graph data for state 0: {dataset_109[0]}")
    '''

    dataset_109_features = features_GraphDataset(root='graph_data', panel_number=103, freq=3)
    print(f"Dataset loaded with {len(dataset_109_features)} states.")
    print(f"Example graph data for state 0: {dataset_109_features[0]}")
