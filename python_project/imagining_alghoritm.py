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

from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import device
import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt

from plot_panel import SENSOR_POSITIONS, SENSOR_PAIRS, PANEL_H, PANEL_W, _draw_static_panel
from graph_dataset import Panel_GraphDataset

T = 0.001 * 10e-3
v = 62500
MAX_DIST = v * T

def P(P_arr, grid, state, dataset, model, beta=None):
    X, Y = grid                          # both shape (100, 100)
    for path_idx, (a, b) in enumerate(SENSOR_PAIRS):
        sHI = extract_sHI_after_GAN(model, dataset, state, path_idx)
        p1 = SENSOR_POSITIONS[a - 1]    # shape (2,)
        p2 = SENSOR_POSITIONS[b - 1]    # shape (2,)

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)   # shape (100, 100)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)   # shape (100, 100)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # scalar

        R = (d1 + d2) / d - 1                            # shape (100, 100)

        if beta is None:
            beta = optimal_beta(d, MAX_DIST)
            print(f"Path {path_idx}: d={d:.2f}, beta={beta:.2f}")

        W = np.where(R < beta, 1 - R / beta, 0.0)        # shape (100, 100)

        P_arr += sHI * W                                 # accumulate in-place

def U(U_arr, grid, beta=None):
    X, Y = grid
    
    for path_idx, (a, b) in enumerate(SENSOR_PAIRS):
        p1 = SENSOR_POSITIONS[a - 1]
        p2 = SENSOR_POSITIONS[b - 1]

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        R = (d1 + d2) / d - 1
        if beta is None:
            beta = optimal_beta(d, MAX_DIST)
            print(f"Path {path_idx}: d={d:.2f}, beta={beta:.2f}")
        
        W = np.where(R < beta, 1 - R / beta, 0.0)

        U_arr += W

    

def R_THR(U, thr, grid):
    X, Y = grid
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    Area = 0
    total_area = X.shape[0] * X.shape[1] * dx * dy 
    peak_U = max(U.flatten())
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if U[i, j] >= thr * peak_U:
                Area += dx * dy
    
    return Area / total_area

def WCPDI(P, U, c, grid):
    X, Y = grid
    THR = find_threshold(U, c, (X, Y))
    WCPDI_map = np.zeros_like(P)
    peak_U = max(U.flatten())
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > THR * peak_U:
                WCPDI_map[i, j] = (P[i, j] - THR * peak_U) / (U[i, j]/peak_U)
            else:
                WCPDI_map[i, j] = 0.0

    return WCPDI_map

def find_threshold(U, c, grid): 
    # find where R_THR = c
    peak_U = max(U.flatten())
    for thr in np.linspace(0, 1, 100):
        if R_THR(U, thr, grid) >= c:
            return thr
    return 0.0

def extract_sHI_after_GAN(model, dataset, state, path_index, extract_HI=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    HIs = []
    states_idx = []
    with torch.no_grad():
        for data in loader:
            if data.y.item() != state and not extract_HI:
                continue    
            data = data.to(device)
            out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            
            if not extract_HI:
                if (out == 0).all():
                    print(f"Warning: Model output is all zeros for state {state}. Check if the model is producing correct outputs.")
                if data.y.item() == state:
                    return out[path_index].item()
            else:
                HIs.append(HI.item())
                states_idx.append(data.y.item())
        return HIs, states_idx
    
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



def animate_panel(panel_number, model,n_pixels, c, beta=None, state_to_show=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # find dataset
    dataset = Panel_GraphDataset(root='graph_data', panel_number=panel_number, freq=3, big_latent=True)
    n_states = len(dataset)
    model = model.to(device)

    dA = (PANEL_W*PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (n_x, n_y  )

    U_arr = np.zeros_like(X)
    U(U_arr, (X, Y), beta)

    fig, ax = plt.subplots(figsize=(6, 8))
    _draw_static_panel(ax, panel_number)

    frames = []
    for state in range(n_states):
        P_arr = np.zeros_like(X)
        P(P_arr, (X, Y), state, dataset, model, beta)
        WCPDI_map = WCPDI(P_arr, U_arr, c, (X, Y))
        im = ax.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=0)
        title = ax.text(0.5, 1.01, f"State {state}/{n_states - 1}",
                        transform=ax.transAxes, ha='center', va='bottom')
        frames.append([im, title])
        if state == state_to_show:
            fig_snap, ax_snap = plt.subplots(figsize=(6, 8))
            _draw_static_panel(ax_snap, panel_number)
            im_snap = ax_snap.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot')
            divider = make_axes_locatable(ax_snap)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig_snap.colorbar(im_snap, cax=cax, label='WCPDI Value')
            ax_snap.set_title(f'WCPDI Map for State {state}')
            ax_snap.set_xlabel('X Position')
            ax_snap.set_ylabel('Y Position')
            fig_snap.tight_layout()
            fig_snap.savefig(f"WCPDI_map_state_{state}.svg", format='svg', bbox_inches='tight')
            plt.show()
            

    anim = animation.ArtistAnimation(fig, frames, interval=200, blit=True)

    anim.save(f'panel_{panel_number}_WCPDI_animation.gif', writer='pillow')


def optimal_beta(d, max_d):
    if d >= max_d:
        return 0.0
    else:
        return   max_d/d-1
            
if __name__ == "__main__":
    dir = r"GCN_models\gcn_big_latent.pt"
    model = torch.load(dir, weights_only=False)
    dataset_109 = Panel_GraphDataset(root='graph_data', panel_number=109, freq=3, big_latent=True)
    dataset_104 = Panel_GraphDataset(root='graph_data', panel_number=104, freq=3, big_latent=True)
    dataset_123 = Panel_GraphDataset(root='graph_data', panel_number=123, freq=3, big_latent=True)

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
    plt.imshow(P_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot')
    plt.colorbar(label='Damage Probability')
    plt.title('Damage Probability Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
   

    U_values = np.zeros((len(x), len(y)))
    U(U_values, grid)

    plt.figure(figsize=(6, 8))
    plt.imshow(U_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot')
    plt.colorbar(label='Weight Sum')
    plt.title('Weight Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
    
    WCPDI_map = WCPDI(P_values, U_values, c=0.9, grid=grid)

    plt.figure(figsize=(6, 8))
    plt.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot')
    plt.colorbar(label='WCPDI Value')
    plt.title('WCPDI Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    plt.tight_layout()
   
    plt.show()

    animate_panel(panel_number=123, model=model, n_pixels=10000, c=0.9, beta=beta, state_to_show=60)
