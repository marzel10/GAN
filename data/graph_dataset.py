'''
This file:
- defines the Panel_GraphDataset class for creating graph datasets from the extracted sHI values and attention-based adjacency matrices
- the dataset is created for each state, where nodes represent paths and edges represent crossing paths weighted by the attention values
- includes a main block that demonstrates how to load the dataset and print some example graph data for a given panel and frequency
- by default, it can load either the sHI features or the big latent features extracted from the autoencoder, depending on the big_latent flag

If you do any changes in the process method, make sure to delete the processed files to trigger re-processing with the new code.
'''

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import warnings
# format error messages with panel and state information for easier debugging
warnings.formatwarning = lambda msg, *_, **__: f"Warning: {msg}\n"

from torch_geometric.data import Data, InMemoryDataset
import torch
from weight_matrix import find_crossings, adjencency_matrix, find_crossings_by_area
import numpy as np
from create_datastores import states
from config import BETA_CONSTANT, NR_SAVED_STATES, STATE_START_INDICES, mat_file_path,  PANEL_123_SUBPANELS

class Panel_GraphDataset(InMemoryDataset):
    def __init__(self, root, panel_number,freq, big_latent=True, type="basic", transform=None, pre_transform=None, beta_constant=BETA_CONSTANT):
        self.panel_number = panel_number
        self.freq = freq
        self.big_latent = big_latent # if True, the model is trained on the latent space of AE, otherwise on the sub-HI (one number per node)
        self.type = type
        self.beta_constant = beta_constant
        super().__init__(root, transform, pre_transform)
        
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"panel_{self.panel_number}_shi_raw_{self.freq}.pt"]

    
    @property
    def processed_file_names(self):
        if self.big_latent:
            if self.type == "basic" or self.type == "without_map_loss":
                return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}.pt"]
            elif self.type == "by_area":
                if self.beta_constant!=BETA_CONSTANT:
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_area_beta{self.beta_constant}.pt"]
                return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_area.pt"]
            elif self.type == "geometry":
                return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_geom.pt"]
            elif self.type == "peak":
                return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_peak.pt"]
            else:
                raise ValueError(f"Unknown adjacency matrix type: {self.type}")

        else:
            if self.type == "basic" or self.type == "without_map_loss":
                return [f"panel_{self.panel_number}_processed_{self.freq}.pt"]
            elif self.type == "by_area":
                if self.beta_constant!=BETA_CONSTANT:
                    return [f"panel_{self.panel_number}_processed_{self.freq}_area_beta{self.beta_constant}.pt"]
                return [f"panel_{self.panel_number}_processed_{self.freq}_area.pt"]
            elif self.type == "geometry":
                return [f"panel_{self.panel_number}_processed_{self.freq}_geom.pt"]
            elif self.type == "peak":
                return [f"panel_{self.panel_number}_processed_{self.freq}_peak.pt"]
            else:
                raise ValueError(f"Unknown adjacency matrix type: {self.type}")
 
    

    def process(self):
        # Load sub_HI from AE
        raw_data = torch.load(self.raw_paths[0], weights_only=False)
        
        features = np.array(raw_data["shi"]) # shape: paths x states x 1
        big_latent = raw_data["big_latent"] # shape: paths x states x latent_dim

        is_123 = str(self.panel_number) == "123"
        subpanel_states_cache = {} if is_123 else None
        if not is_123:
            current_state = states(str(mat_file_path(self.panel_number)))

        # Process raw_data into a list of Data objects
        data_list = []
        t_begin = time.perf_counter()
        for state in range(features.shape[1]):
            if self.big_latent:
                x = torch.tensor(big_latent[:, state, :], dtype=torch.float)  # shape: (num_paths, latent_dim)
            else:
                x = torch.tensor(features[:, state, 0], dtype=torch.float).unsqueeze(1)  # shape: (num_paths, 1)

            if (x == 0).all():
                warnings.warn(f"Warning: All features are zero for state {state}. Check if the raw features are correct.")

            connection_matrix = find_crossings_by_area() if type == "by_area" else find_crossings()
            edge_index = torch.tensor(np.array(np.nonzero(connection_matrix)), dtype=torch.long)  # shape: 2 x num_edges

            t_state_load_start = time.perf_counter()
            if is_123:
                    # find which subpanel this state belongs to
                    subpanel = None
                    for sp in NR_SAVED_STATES.keys():
                        start_idx = STATE_START_INDICES[sp]
                        end_idx = start_idx + NR_SAVED_STATES[sp]
                        if start_idx <= state < end_idx:
                            subpanel = sp
                            break
                    if subpanel is None:
                        raise ValueError(f"State index {state} does not belong to any known subpanel.")
                    if subpanel not in subpanel_states_cache:
                        subpanel_states_cache[subpanel] = states(str(mat_file_path(subpanel)))
                    current_state = subpanel_states_cache[subpanel]
            t_state_load_end = time.perf_counter()

            adj_matrix = adjencency_matrix(current_state, state_idx=state, freq_idx=self.freq, type=self.type, beta_constant=self.beta_constant)  # shape: paths x paths
            t_adj_end = time.perf_counter()

            if (adj_matrix == 0).all():
                warnings.warn(f"Warning: Adjacency matrix for state {state} is all zeros. Check if the attention values are correct.")


            adj_matrix = adj_matrix[connection_matrix != 0]  # shape: num_edges
            edge_weight = torch.tensor((np.array(adj_matrix)), dtype=torch.float)


            data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                                  y=torch.tensor([state], dtype=torch.long),
                                  panel=torch.tensor([self.panel_number], dtype=torch.long)))
            t_after_append = time.perf_counter()
            print(f"state {state}: states()={t_state_load_end - t_state_load_start:.3f}s "
                  f"adjencency_matrix={t_adj_end - t_state_load_end:.3f}s "
                  f"append={t_after_append - t_adj_end:.3f}s "
                  f"total_elapsed={t_after_append - t_begin:.2f}s")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class features_GraphDataset(InMemoryDataset):
    # This dataset is used only if the graph is trained on raw time-frequency features. No adjencncy matrix type implemented
    def __init__(self, root, panel_number,freq, transform=None, pre_transform=None):
        self.panel_number = panel_number
        self.freq = freq
        
        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.panel_number == 123:
            return [str(f"States_{sp}_freq{self.freq}_all_paths_diff_full.npy") for sp in PANEL_123_SUBPANELS]
        else:
            return [str( f"States_{self.panel_number}_freq{self.freq}_all_paths_diff_full.npy")]
    
    @property
    def processed_file_names(self):
        return [f"panel_{self.panel_number}_processed_raw_features_diff_{self.freq}.pt"]
    
    def process(self):
        if self.panel_number == 123:
            features_list = []
            for file in self.raw_paths:
                features = np.load(file)
                features_list.append(features)
            features = np.concatenate(features_list, axis=1)  # shape: paths x states x features
            print(f"Concatenated features shape for panel 123: {features.shape}")
        else:
            features = np.load(self.raw_paths[0])  # shape: paths x states x features
            print(f"Loaded features shape for panel {self.panel_number}: {features.shape}")

        # 
        is_123 = str(self.panel_number) == "123"
        subpanel_states_cache = {} if is_123 else None
        if not is_123:
            current_state = states(str(mat_file_path(self.panel_number)))

        # Process features into a list of Data objects
        data_list = []
        
        for state in range(features.shape[1]):
            x = torch.tensor(features[:, state, :], dtype=torch.float)  # shape: (num_paths, num_features)

            # Create edge weights
            connection_matrix = find_crossings()
            edge_index = torch.tensor(np.array(np.nonzero(connection_matrix)), dtype=torch.long)
            if is_123:
                    # find which subpanel this state belongs to
                    subpanel = None
                    for sp in NR_SAVED_STATES.keys():
                        start_idx = STATE_START_INDICES[sp]
                        end_idx = start_idx + NR_SAVED_STATES[sp]
                        if start_idx <= state < end_idx:
                            subpanel = sp
                            break
                    if subpanel is None:
                        raise ValueError(f"State index {state} does not belong to any known subpanel.")
                    if subpanel not in subpanel_states_cache:
                        subpanel_states_cache[subpanel] = states(str(mat_file_path(subpanel)))
                    current_state = subpanel_states_cache[subpanel]

            adj_matrix = adjencency_matrix(current_state, state_idx=state, freq_idx=self.freq)  # shape: paths x paths
            adj_matrix = adj_matrix[connection_matrix != 0]  # shape: num_edges
            edge_weight = torch.tensor((np.array(adj_matrix)), dtype=torch.float)
            data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                                  y=torch.tensor([state], dtype=torch.long),
                                  panel=torch.tensor([self.panel_number], dtype=torch.long)))
        data, slices = self.collate(data_list)

        # make sure the processed directory exists
        processed_dir = Path(self.processed_paths[0]).parent
        processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])     
        print(f"Processed data saved to {self.processed_paths[0]} with {len(data_list)} states.")  


