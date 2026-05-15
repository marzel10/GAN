'''
This code implementes the WCPDI alghoritm from 
"A novel probability-based diagnostic imaging with weight compensation for
damage localization using guided waves"
'''

from matplotlib import animation
from torch import device
import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt

from plot_panel import SENSOR_POSITIONS, SENSOR_PAIRS, PANEL_H, PANEL_W, _draw_static_panel
from graph_dataset import Panel_GraphDataset

def P(P_arr, grid, beta, state, dataset, model):
    X, Y = grid                          # both shape (100, 100)
    for path_idx, (a, b) in enumerate(SENSOR_PAIRS):
        sHI = extract_sHI_after_GAN(model, dataset, state, path_idx)
        p1 = SENSOR_POSITIONS[a - 1]    # shape (2,)
        p2 = SENSOR_POSITIONS[b - 1]    # shape (2,)

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)   # shape (100, 100)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)   # shape (100, 100)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # scalar

        R = (d1 + d2) / d - 1                            # shape (100, 100)
        W = np.where(R < beta, 1 - R / beta, 0.0)        # shape (100, 100)

        P_arr += sHI * W                                 # accumulate in-place

def U(U_arr, grid, beta):
    X, Y = grid
    
    for path_idx, (a, b) in enumerate(SENSOR_PAIRS):
        p1 = SENSOR_POSITIONS[a - 1]
        p2 = SENSOR_POSITIONS[b - 1]

        d1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2)
        d2 = np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)
        d  = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        R = (d1 + d2) / d - 1
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

def find_threshold(U, c, grid): # CHECK IMPLEMENTATION, THIS IS A ROUGH APPROXIMATION
    # find where R_THR = c
    peak_U = max(U.flatten())
    for thr in np.linspace(0, 1, 100):
        if R_THR(U, thr, grid) >= c:
            return thr
    return 0.0

def extract_sHI_after_GAN(model, dataset, state, path_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for data in loader:
            if data.y.item() != state:
                continue    
            data = data.to(device)
            out, HI = model(data.x, data.edge_index, data.batch)
            
            if (out == 0).all():
                print(f"Warning: Model output is all zeros for state {state}. Check if the model is producing correct outputs.")
            if data.y.item() == state:
                return out[path_index].item()
            

def animate_panel(panel_number, model,n_pixels, beta, c):

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
        P(P_arr, (X, Y), beta, state, dataset, model)
        WCPDI_map = WCPDI(P_arr, U_arr, c, (X, Y))
        im = ax.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=0)
        title = ax.text(0.5, 1.01, f"State {state}/{n_states - 1}",
                        transform=ax.transAxes, ha='center', va='bottom')
        frames.append([im, title])

    anim = animation.ArtistAnimation(fig, frames, interval=200, blit=True)

    anim.save(f'panel_{panel_number}_WCPDI_animation.gif', writer='pillow')



            
if __name__ == "__main__":
    dir = r"GCN_models\gcn_big_latent.pt"
    model = torch.load(dir, weights_only=False)
    dataset_109 = Panel_GraphDataset(root='graph_data', panel_number=109, freq=3, big_latent=True)
    
    sHi = extract_sHI_after_GAN(model, dataset_109, state=0, path_index=0)
    print(f"Extracted sHI: {sHi}")

    beta = 0.7

    x = np.linspace(0, PANEL_W, 100)
    y = np.linspace(0, PANEL_H, 100)
    P_values = np.zeros((len(x), len(y)))
    grid = np.meshgrid(x, y, indexing='ij')

    P(P_values, grid, beta, state=0, dataset=dataset_109, model=model)  

    plt.figure(figsize=(6,8))
    plt.imshow(P_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot')
    plt.colorbar(label='Damage Probability')
    plt.title('Damage Probability Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
   

    U_values = np.zeros((len(x), len(y)))
    U(U_values, grid, beta)

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
   


    animate_panel(panel_number=123, model=model, n_pixels=10000, beta=0.7, c=0.9)
