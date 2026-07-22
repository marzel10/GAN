'''
This code implementes the WCPDI alghoritm from 
"A novel probability-based diagnostic imaging with weight compensation for
damage localization using guided waves"

important functions:
- P(P_arr, grid, beta, state, dataset, model): computes the damage probability map P for a given state using the sHI values from the GCN model and the spatial weighting function based on the sensor paths
- U(U_arr, grid, beta): computes the weight map U based on the sensor paths and the spatial weighting function (like P but without the sHI values)
- R_THR(U, thr, grid): computes the area ratio R for a given weight map U and threshold thr (how much of the area has weight above the threshold, how active is the whole panel)
- WCPDI(P, U, c, grid): computes the WCPDI map by applying a threshold to P based on the area ratio of U and normalizing by U
- find_threshold(U, c, grid): finds the threshold for P based on the area ratio of U and the desired coverage c (what percentage of the panel should be considered active based on U)
- extract_sHI_after_GAN(model, dataset, state, path_index): extracts the sHI value for a specific state and path from the trained GCN model
- animate_panel(panel_number, model, n_pixels, beta, c): creates an animation of the WCPDI map over the states for a given panel using the trained GCN model
'''

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os

from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import device
import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt

from plot_panel import SENSOR_POSITIONS, SENSOR_PAIRS, PANEL_H, PANEL_W, DAMAGE_POINTS, _draw_static_panel
from graph_dataset import Panel_GraphDataset, features_GraphDataset
from extract_shi import extract_shi

from weight_matrix import failed_sensor_at
from config import (
    MAX_DIST, GRAPH_DATA_DIR, ANIMATIONS_DIR, GCN_MODELS_DIR, DEFAULT_FREQ_INDEX,
    DEFAULT_WCPDI_C, DEFAULT_N_PIXELS, CMAP_HEATMAP,
)

def P(P_arr, grid, state, dataset, model, beta=None):
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

        path_beta = beta if beta is not None else optimal_beta(d, MAX_DIST)

        W = np.where(R < path_beta, 1 - R / path_beta, 0.0)   # shape (100, 100)

        P_arr += sHI * W                                 # accumulate in-place

def P_AE(P_arr, grid, sHI_per_state):
    X, Y = grid                          # both shape (100, 100)
    for path_idx, (a, b) in enumerate(SENSOR_PAIRS):
        sHI = sHI_per_state[path_idx]     # sHI[0]: array (n_states, 1) for this path
        p1 = SENSOR_POSITIONS[a - 1]    # shape (2,)
        p2 = SENSOR_POSITIONS[b - 1]    # shape (2,)

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)   # shape (100, 100)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)   # shape (100, 100)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # scalar

        R = (d1 + d2) / d - 1                            # shape (100, 100)

        path_beta = optimal_beta(d, MAX_DIST)

        W = np.where(R < path_beta, 1 - R / path_beta, 0.0)  # shape (100, 100)

        P_arr += sHI * W                                 # accumulate in-place

def U(U_arr, grid, beta=None, panel_number=None, state=None, path_indices=None):
    """path_indices: 0-indexed indices into SENSOR_PAIRS to restrict the sum to
    (None = every path, the original behavior -- used by e.g. make_elipses.py to
    isolate a single path's ellipse of influence)."""
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

        path_beta = beta if beta is not None else optimal_beta(d, MAX_DIST)

        W = np.where(R < path_beta, 1 - R / path_beta, 0.0)

        U_arr += W

    

def R_THR(U, thr, grid):
    X, Y = grid
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    total_area = X.shape[0] * X.shape[1] * dx * dy
    peak_U = U.max()
    active_pixels = np.sum(U >= thr * peak_U)
    return active_pixels * dx * dy / total_area

def find_threshold(U, c, grid):
    X = grid[0]
    total_pixels = X.shape[0] * X.shape[1]
    peak_U = U.max()
    # for each candidate threshold, count active pixels in one vectorised pass
    thresholds = np.linspace(0, 1, 100)
    for thr in thresholds:
        active = np.sum(U >= thr * peak_U)
        if active / total_pixels >= c:
            return thr
    return 0.0

def WCPDI(P, U):
    peak_U = U.max()


    # Pixels with no surviving sensor-path coverage (U == 0, e.g. near a
    # failed sensor once its paths are excluded) have no defined WCPDI value
    # -- mark them nan explicitly instead of leaving it to an incidental 0/0.
    valid = U > 0
    #WCPDI_map = np.full_like(P, np.nan)
    # with np.errstate(invalid='ignore', divide='ignore'):
    #     WCPDI_map[valid] = P[valid] / (U[valid] / peak_U)
    WCPDI_map = P/U*peak_U
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
    


    
def plot_HI(model, dataset, title="HI vs State Index", save_path=None):
    HI, states_idx = extract_sHI_after_GAN(model, dataset, state=0, path_index=1, extract_HI=True)

    plt.figure(figsize=(6,6))
    plt.scatter(states_idx, HI)
    plt.xlabel("State index")
    plt.ylabel("HI")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()



def _safe_minmax(m):
    """nanmin/nanmax that degrade to (0.0, 1.0) instead of nan when a frame
    has no valid (non-blind) pixels left, e.g. once enough sensors have
    failed that a state's WCPDI map is all-nan."""
    if np.all(np.isnan(m)):
        return 0.0, 1.0
    return np.nanmin(m), np.nanmax(m)


def animate_panel_sidebyside(panel_number, model, n_pixels, c, beta=None, features=False, transform=None, output_dir=str(ANIMATIONS_DIR), file_name=None, faults_data=None):
    """Two-panel animation.

    Left:  globally normalised WCPDI (0-1 scale fixed across all states).
    Right: per-frame adaptive scale — the full colour range always spans the
           current frame's [min, max], so the colorbar values change each frame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if features:
        dataset = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel_number, freq=DEFAULT_FREQ_INDEX)
    else:
        dataset = Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel_number, freq=DEFAULT_FREQ_INDEX, big_latent=True)
    if transform is not None:
        dataset.transform = transform
    n_states = len(dataset)
    model = model.to(device)

    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')

    wcpdi_maps = []
    for state in range(n_states):
        U_arr = np.zeros_like(X)
        U(U_arr, (X, Y), beta, panel_number=dataset.panel_number, state=state)
        P_arr = np.zeros_like(X)
        P(P_arr, (X, Y), state, dataset, model, beta)
        wcpdi_maps.append(WCPDI(P_arr, U_arr, c, (X, Y)))

    # WCPDI at the known damage point vs. the panel-wide mean, tracked across states.
    damage_point = DAMAGE_POINTS[panel_number]
    if damage_point.ndim == 2:
        damage_point = damage_point.mean(axis=0)
    ix = int(np.clip(round(damage_point[0] / dx), 0, X.shape[0] - 1))
    iy = int(np.clip(round(damage_point[1] / dx), 0, X.shape[1] - 1))
    damage_vals = np.array([m[ix, iy] for m in wcpdi_maps])
    mean_vals = np.array([
        np.nanmean(m) if not np.all(np.isnan(m)) else np.nan
        for m in wcpdi_maps
    ])

    vmin_global, vmax_global = _safe_minmax(np.stack(wcpdi_maps))
    span_global = (vmax_global - vmin_global) or 1.0

    def _norm(m):
        return (m - vmin_global) / span_global

    fig, (ax_norm, ax_adapt, ax_line) = plt.subplots(1, 3, figsize=(18, 8))
    _draw_static_panel(ax_norm, panel_number)
    _draw_static_panel(ax_adapt, panel_number)

    divider_norm = make_axes_locatable(ax_norm)
    cax_norm = divider_norm.append_axes("right", size="5%", pad=0.05)
    im_norm = ax_norm.imshow(_norm(wcpdi_maps[0]).T, extent=(0, PANEL_W, 0, PANEL_H),
                             origin='lower', cmap=CMAP_HEATMAP, vmin=0, vmax=1)
    cbar_norm = fig.colorbar(im_norm, cax=cax_norm, label='Normalised WCPDI')
    ax_norm.set_title('Normalised WCPDI')
    ax_norm.set_xlabel('X Position')
    ax_norm.set_ylabel('Y Position')

    m0 = wcpdi_maps[0]
    fmin0, fmax0 = _safe_minmax(m0)
    if fmax0 == fmin0:
        fmax0 = fmin0 + 1.0
    divider_adapt = make_axes_locatable(ax_adapt)
    cax_adapt = divider_adapt.append_axes("right", size="5%", pad=0.05)
    im_adapt = ax_adapt.imshow(m0.T, extent=(0, PANEL_W, 0, PANEL_H),
                               origin='lower', cmap=CMAP_HEATMAP, vmin=fmin0, vmax=fmax0)
    cbar_adapt = fig.colorbar(im_adapt, cax=cax_adapt, label='WCPDI Value')
    ax_adapt.set_title('Adaptive-scale WCPDI')
    ax_adapt.set_xlabel('X Position')
    ax_adapt.set_ylabel('Y Position')

    title = ax_norm.text(0.5, 1.05, f"State 0/{n_states - 1}",
                         transform=ax_norm.transAxes, ha='center', va='bottom')

    ax_line.plot(range(n_states), damage_vals, label='WCPDI at damage point',
                 color='crimson', marker='o', markersize=3)
    ax_line.plot(range(n_states), mean_vals, label='Mean WCPDI (panel)',
                 color='steelblue', marker='o', markersize=3)
    cursor_line = ax_line.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax_line.set_xlabel('State')
    ax_line.set_ylabel('WCPDI Value')
    ax_line.set_title('WCPDI at Damage Point vs Mean')
    ax_line.legend()
    ax_line.grid(True)

    fig.tight_layout()

    def update(state):
        m = wcpdi_maps[state]
        im_norm.set_data(_norm(m).T)

        fmin, fmax = _safe_minmax(m)
        if fmax == fmin:
            fmax = fmin + 1.0
        im_adapt.set_data(m.T)
        im_adapt.set_clim(fmin, fmax)
        cbar_adapt.update_normal(im_adapt)

        cursor_line.set_xdata([state, state])

        title.set_text(f"State {state}/{n_states - 1}")
        return [im_norm, im_adapt, title, cursor_line]

    anim = animation.FuncAnimation(fig, update, frames=n_states, interval=200, blit=False)

    if file_name is None:
        file_name = f"WCPDI_panel_{panel_number}_sidebyside"
    anim.save(f'{output_dir}/{file_name}_animation.gif', writer='pillow')
    plt.close(fig)


def optimal_beta(d, max_d):
    if d >= max_d:
        return 0.0
    else:
        return   (max_d/d-1)/50
            
if __name__ == "__main__":
    features=True
    f = DEFAULT_FREQ_INDEX
    if features:
        dir = str(GCN_MODELS_DIR / "gcn_features_lin.pt")

        dataset_103 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=103, freq=f)
        dataset_104 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=104, freq=f)
        dataset_105 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=105, freq=f)
        dataset_109 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=109, freq=f)
        dataset_123 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=123, freq=f)
        bundle = torch.load(dir, weights_only=False)
        if isinstance(bundle, dict):
            model = bundle['model']
            feature_mean = bundle['feature_mean']
            feature_std  = bundle['feature_std']
            def norm_transform(data):
                data.x = (data.x - feature_mean) / feature_std
                return data
            for ds in [dataset_103, dataset_104, dataset_105, dataset_109, dataset_123]:
                ds.transform = norm_transform
        else:
            model = bundle
            norm_transform = None
    else:
        dir = str(GCN_MODELS_DIR / "gcn_big_latent.pt")
    
        model = torch.load(dir, weights_only=False)
        dataset_109 = Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=109, freq=f, big_latent=True)
        dataset_104 = Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=104, freq=f, big_latent=True)
        dataset_123 = Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=123, freq=f, big_latent=True)

    sHi = extract_sHI_after_GAN(model, dataset_109, state=0, path_index=0)
    print(f"Extracted sHI: {sHi}")

    plot_HI(model, dataset_109, title="HI vs State Index for validation panel 109", save_path="HI_vs_state_109_val.svg")
    plot_HI(model, dataset_104, title="HI vs State Index for training panel 104", save_path="HI_vs_state_104_train.svg")
    plot_HI(model, dataset_123, title="HI vs State Index for test panel 123", save_path="HI_vs_state_123_test.svg")
    beta = 0.7


    x = np.linspace(0, PANEL_W, 100)
    y = np.linspace(0, PANEL_H, 100)
    P_values = np.zeros((len(x), len(y)))
    grid = np.meshgrid(x, y, indexing='ij')

    P(P_values, grid, state=0, dataset=dataset_109, model=model)  

    plt.figure(figsize=(6,8))
    plt.imshow(P_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='Damage Probability')
    plt.title('Damage Probability Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
   

    U_values = np.zeros((len(x), len(y)))
    U(U_values, grid, panel_number=dataset_109.panel_number, state=0)

    plt.figure(figsize=(6, 8))
    plt.imshow(U_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='Weight Sum')
    plt.title('Weight Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
    
    WCPDI_map = WCPDI(P_values, U_values, c=DEFAULT_WCPDI_C, grid=grid)

    plt.figure(figsize=(6, 8))
    plt.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='WCPDI Value')
    plt.title('WCPDI Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    plt.tight_layout()
   
    plt.show()

    animate_panel_sidebyside(panel_number=103, model=model, n_pixels=DEFAULT_N_PIXELS, c=DEFAULT_WCPDI_C, beta=beta, state_to_show=0, features=features, transform=norm_transform if features else None)
