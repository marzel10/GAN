import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch_geometric
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.signal import resample_poly

from graph_dataset import Panel_GraphDataset, features_GraphDataset
from extract_shi import extract_shi
from config import GRAPH_DATA_DIR, DEFAULT_FREQ_INDEX, MODELS_FEATURES_DIR


def prepare_DI_from_AE(panels, folder=MODELS_FEATURES_DIR):
    DI_paths = [] # storage for the DI values for every path in every panel (28xnr_panelsx nr_states)
    DI = [] # storage for the DI values for every panel (nr_panelsx nr_states)

    for panel in panels:
        all_latents, _, _ = extract_shi([folder], DEFAULT_FREQ_INDEX, panel, GAN_dir=str(folder), features=True)

        # Each path's latent is (n_states, 1) -- squeeze to (n_states,) and
        # stack across paths into a (28, n_states) matrix for this panel.
        path_di = np.stack([np.asarray(latent).reshape(-1) for latent in all_latents], axis=0)

        DI_paths.append(path_di)
        DI.append(np.mean(path_di, axis=0))  # average over paths -> (n_states,)

    return DI, DI_paths

def prepare_DI_from_graph(panels, model, features=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if features:
        datasets = [features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=DEFAULT_FREQ_INDEX) for panel in panels]
    else:
        datasets = [Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=DEFAULT_FREQ_INDEX, big_latent=True) for panel in panels]
    
    model.eval()
    M = len(panels)
    DI = []
    DI_single = []
    for dataset in datasets:
        loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                DI_single.append(HI.item())
        DI.append(DI_single)

    
    return DI

def monotonicity_criterion(DI):
    # Check if DI is monotonically increasing across states for each panel
    M = len(DI)
    monotonicity=0
    
    for i in range(M):
        panel_monotonicity  = 0
        panel_DI = DI[i]
        N = len(panel_DI)
        
        for j in range(N):
            partial_panel_monotonicity  = 0
            sum_weights = 0

            for k in range(j+1, N):
                point_mono = np.sign(panel_DI[j] - panel_DI[k])*(k-j)
                partial_panel_monotonicity += point_mono
                sum_weights += (k-j)
            
            panel_monotonicity += partial_panel_monotonicity / sum_weights if sum_weights != 0 else 0

        panel_monotonicity = np.abs(panel_monotonicity/(N-1))
        monotonicity += panel_monotonicity

    return monotonicity / M
           



def trendability_criterion(DI):

    M = len(DI)
    trendability_matrix = np.zeros((M, M))

    for i in range(M):
        vector_i = np.asarray(DI[i])
        for j in range(M):
            vector_j = np.asarray(DI[j])

            # Resample the shorter vector up to the longer one's length so
            # pearsonr can compare panels with different state counts
            # (e.g. panel 103 has 32 states, 104 has 58).
            if len(vector_i) != len(vector_j):
                if len(vector_i) < len(vector_j):
                    v_i = resample_poly(vector_i, len(vector_j), len(vector_i), window=('kaiser', 5))
                    v_j = vector_j
                else:
                    v_i = vector_i
                    v_j = resample_poly(vector_j, len(vector_i), len(vector_j), window=('kaiser', 5))
            else:
                v_i, v_j = vector_i, vector_j

            trendability_matrix[i, j] = abs(pearsonr(v_i, v_j)[0])

    return np.min(trendability_matrix)

def prognosability_criterion(DI):

    M = len(DI)
    prognosability = 0

    end_point_std = 0
    mean_HI_range = 0

    mean_end_value = 0

    for j in range(M):    
        mean_end_value += DI[j][-1]
    mean_end_value /= M

    for i in range(M):
        end_point_std += np.abs(DI[i][-1] - mean_end_value)**2
        mean_HI_range += np.abs(DI[i][-1] - DI[i][0])

    mean_HI_range /= M
    end_point_std = np.sqrt(end_point_std / M)

    prognosability = np.exp(-end_point_std / mean_HI_range)

    return prognosability

def fitness_criterion(panels, model, a=1, b=1, c=1, graph = True):
    if graph:
        DI = prepare_DI_from_graph(panels, model)
    else:
        DI, _ = prepare_DI_from_AE(panels)
    monotonicity = monotonicity_criterion(DI)
    trendability = trendability_criterion(DI)
    prognosability = prognosability_criterion(DI)

    # Combine criteria into a single fitness score (weights can be adjusted)
    fitness_score = a * monotonicity + b * trendability + c * prognosability
    return fitness_score