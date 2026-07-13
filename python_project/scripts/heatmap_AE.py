# This code generates heatmap on the panel using sHI from the autoencoder

from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt

from imagining_alghoritm import P_AE, U, WCPDI
from extract_shi import extract_shi
from plot_panel import PANEL_H, PANEL_W, _draw_static_panel


def animate_panel_AE(panel_number, folders, freq, n_pixels, c, beta=None, state_to_show=None, features=False):
    dataset = str(panel_number)

    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = (X, Y)

    U_arr = np.zeros_like(X)
    U(U_arr, grid, beta)

    sHI_all, _, _ = extract_shi(folders, freq, dataset, features=features)
    n_states = sHI_all[0].shape[0]

    # Pre-compute all maps to get a consistent colorbar scale across frames
    wcpdi_maps = []
    for state in range(n_states):
        P_arr = np.zeros_like(X)
        sHI_per_state = [sHI[state, 0] for sHI in sHI_all]
        P_AE(P_arr, grid, sHI_per_state, folders, freq, dataset, features=features, beta=beta)
        wcpdi_maps.append(WCPDI(P_arr, U_arr, c, grid))

    vmax = max(m.max() for m in wcpdi_maps)
    vmin = min(m.min() for m in wcpdi_maps)
    if vmax == 0:
        vmax = 1.0

    fig, (ax, ax_norm) = plt.subplots(1, 2, figsize=(12, 8))
    _draw_static_panel(ax, panel_number)
    _draw_static_panel(ax_norm, panel_number)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im0 = ax.imshow(wcpdi_maps[0].T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    fig.colorbar(im0, cax=cax, label='WCPDI Value')
    ax.set_title('WCPDI')

    divider_norm = make_axes_locatable(ax_norm)
    cax_norm = divider_norm.append_axes("right", size="5%", pad=0.05)
    im0_norm = ax_norm.imshow((wcpdi_maps[0] / vmax).T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=0, vmax=1)
    fig.colorbar(im0_norm, cax=cax_norm, label='Normalised WCPDI')
    ax_norm.set_title('Normalised WCPDI')

    frames = []
    for state, WCPDI_map in enumerate(wcpdi_maps):
        im = ax.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
        im_n = ax_norm.imshow((WCPDI_map / vmax).T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=0, vmax=1)
        title = ax.text(0.5, 1.05, f"State {state}/{n_states - 1}",
                        transform=ax.transAxes, ha='center', va='bottom')
        frames.append([im, im_n, title])

        if state == state_to_show:
            fig_snap, (ax_s, ax_sn) = plt.subplots(1, 2, figsize=(12, 8))
            _draw_static_panel(ax_s, panel_number)
            _draw_static_panel(ax_sn, panel_number)

            im_s = ax_s.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
            div_s = make_axes_locatable(ax_s)
            fig_snap.colorbar(im_s, cax=div_s.append_axes("right", size="5%", pad=0.05), label='WCPDI Value')
            ax_s.set_title(f'WCPDI — State {state}')

            im_sn = ax_sn.imshow((WCPDI_map / vmax).T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot', vmin=vmin/vmax, vmax=1)
            div_sn = make_axes_locatable(ax_sn)
            fig_snap.colorbar(im_sn, cax=div_sn.append_axes("right", size="5%", pad=0.05), label='Normalised WCPDI')
            ax_sn.set_title(f'Normalised WCPDI — State {state}')

            for a in (ax_s, ax_sn):
                a.set_xlabel('X Position')
                a.set_ylabel('Y Position')
            fig_snap.tight_layout()
            fig_snap.savefig(f"WCPDI_AE_map_state_{state}.svg", format='svg', bbox_inches='tight')
            plt.show()

    anim = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    anim.save(f'panel_{panel_number}_WCPDI_AE_animation.gif', writer='pillow')


if __name__ == "__main__":
    features = False
    if features:
        folders = ["models_features"]
    else:
        folders = ["Model_for_every_path"]
    freq = 3
    beta = 0.7
    c = 0.9
    panel_number = 123
    dataset = str(panel_number)

    x = np.linspace(0, PANEL_W, 100)
    y = np.linspace(0, PANEL_H, 100)
    grid = np.meshgrid(x, y, indexing='ij')

    P_values = np.zeros((len(x), len(y)))
    sHI_all, _, _ = extract_shi(folders, freq, dataset, features=features)
    sHI_per_state = [sHI[0, 0] for sHI in sHI_all]  # list of sHI values for this state, one per path
    P_AE(P_values, grid, sHI_per_state, folders=folders, freq=freq, dataset=dataset, beta=beta, features=features)

    plt.figure(figsize=(6, 8))
    plt.imshow(P_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot')
    plt.colorbar(label='Damage Probability')
    plt.title('Damage Probability Map (AE)')
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

    WCPDI_map = WCPDI(P_values, U_values, c=c, grid=grid)

    plt.figure(figsize=(6, 8))
    plt.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap='hot')
    plt.colorbar(label='WCPDI Value')
    plt.title('WCPDI Map (AE)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
    plt.show()

    animate_panel_AE(panel_number=panel_number, folders=folders, freq=freq, n_pixels=10000, c=c, beta=beta, state_to_show=0, features=features)
