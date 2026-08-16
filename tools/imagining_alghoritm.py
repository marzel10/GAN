'''
This code implementes the WCPDI alghoritm from 
"A novel probability-based diagnostic imaging with weight compensation for
damage localization using guided waves"

important functions:
- P(P_arr, grid, beta, state, dataset, model): computes the damage probability map P for a given state using the sHI values from the GCN model and the spatial weighting function based on the sensor paths
- U(U_arr, grid, beta): computes the weight map U based on the sensor paths and the spatial weighting function (like P but without the sHI values)
- WCPDI(P, U): computes the WCPDI map by normalizing P by U (scaled by U's peak)
- extract_sHI_after_GAN(model, dataset, state, path_index): extracts the sHI value for a specific state and path from the trained GCN model
'''

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import torch
import torch_geometric
import numpy as np

from weight_matrix import failed_sensor_at
from config import (
    MAX_DIST, BETA_CONSTANT,
    SENSOR_POSITIONS, SENSOR_PAIRS, 
)

def P(P_arr, grid, state, dataset, model, beta_constant=BETA_CONSTANT):
    X, Y = grid                          # both shape (100, 100)
    failed_sensor = failed_sensor_at(dataset.panel_number, state)

    for path_idx, (a, b) in enumerate(SENSOR_PAIRS):
        # skip paths through a sensor that has failed by this state
        if failed_sensor is not None and (a == failed_sensor or b == failed_sensor):
            continue
        sHI = extract_sHI_after_GAN(model, dataset, state, path_idx)
        p1 = SENSOR_POSITIONS[a - 1]    # shape (2,)
        p2 = SENSOR_POSITIONS[b - 1]    # shape (2,)

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)   # shape (100, 100)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)   # shape (100, 100)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # scalar

        R = (d1 + d2) / d - 1                            # shape (100, 100)

        path_beta = optimal_beta(d, MAX_DIST, beta_constant)

        W = np.where(R < path_beta, 1 - R / path_beta, 0.0)   # shape (100, 100)

        P_arr += sHI * W                                 # accumulate in-place

def P_AE(P_arr, grid, sHI_per_path, panel_number=None, state=None, beta_constant=BETA_CONSTANT):
    """Same as P(), but sourced from an already-computed per-path sHI/HI value for one state """
    X, Y = grid                          # both shape (100, 100)
    failed_sensor = failed_sensor_at(panel_number, state) if panel_number is not None and state is not None else None

    for path_idx, (a, b) in enumerate(SENSOR_PAIRS):
        # skip paths through a sensor that has failed by this state
        if failed_sensor is not None and (a == failed_sensor or b == failed_sensor):
            continue
        sHI = sHI_per_path[path_idx]    # scalar sHI/HI value for this path at this state
        p1 = SENSOR_POSITIONS[a - 1]    # shape (2,)
        p2 = SENSOR_POSITIONS[b - 1]    # shape (2,)

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)   # shape (100, 100)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)   # shape (100, 100)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # scalar

        R = (d1 + d2) / d - 1                            # shape (100, 100)

        path_beta = optimal_beta(d, MAX_DIST, beta_constant)

        W = np.where(R < path_beta, 1 - R / path_beta, 0.0)  # shape (100, 100)

        P_arr += sHI * W                                 # accumulate in-place

def U(U_arr, grid, panel_number=None, state=None, path_indices=None, beta_constant=BETA_CONSTANT):
    
    X, Y = grid
    failed_sensor = failed_sensor_at(panel_number, state) if panel_number is not None and state is not None else None

    for idx, (a, b) in enumerate(SENSOR_PAIRS):
        if path_indices is not None and idx not in path_indices:
            continue
        # skip paths through a sensor that has failed by this state
        if failed_sensor is not None and (a == failed_sensor or b == failed_sensor):
            continue
        p1 = SENSOR_POSITIONS[a - 1]
        p2 = SENSOR_POSITIONS[b - 1]

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        R = (d1 + d2) / d - 1

        if beta_constant is None:
            beta_constant = BETA_CONSTANT
            
        path_beta =optimal_beta(d, MAX_DIST, beta_constant=beta_constant)

        W = np.where(R < path_beta, 1 - R / path_beta, 0.0)

        U_arr += W

def WCPDI(P, U):
    peak_U = U.max()
    valid = U > 0
    WCPDI_map = np.full_like(P, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        WCPDI_map[valid] = P[valid] / (U[valid] / peak_U)
    return WCPDI_map

def extract_sHI_after_GAN(model, dataset, state, path_index, extract_HI=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        if extract_HI:
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            data = next(iter(loader)).to(device)
            _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            return HI.cpu().numpy(), data.y.cpu().numpy()
        else:
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
            for data in loader:
                if data.y.item() != state:
                    continue
                data = data.to(device)
                out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                if (out == 0).all():
                    print(f"Warning: Model output is all zeros for state {state}. Check if the model is producing correct outputs.")
                return out[path_index].item()
    

def optimal_beta(d, max_d, beta_constant):
    if d >= max_d:
        return 0.0
    else:
        return   (max_d/d-1)/beta_constant
            

