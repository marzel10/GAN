# This code generates heatmap on the panel using sHI from the autoencoder

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt

from imagining_alghoritm import P_AE, U, WCPDI, _safe_minmax
from path_performance import load_cached, average_sHI_over_frequency, OUT_DIR, PANELS
from plot_panel import PANEL_H, PANEL_W, _draw_static_panel
from config import DEFAULT_WCPDI_C, DEFAULT_N_PIXELS, CMAP_HEATMAP


def _load_sHI_avg(panel_number, fold, sHI=None):
    '''sHI averaged over the 6 frequencies for every path, from path_performance.py's
    cache. Paths with no data at all (for this panel/fold, across every frequency) fall
    back to a flat-zero contribution, with a printed warning.'''
    dataset = str(panel_number)
    if sHI is None:
        cached = load_cached()
        if cached is None:
            raise FileNotFoundError(
                f"No cached results in {OUT_DIR} -- run path_performance.py first."
            )
        sHI, _ = cached

    sHI_avg = average_sHI_over_frequency(sHI, dataset, fold=fold)
    n_states_seen = {len(c) for c in sHI_avg if c is not None}
    if not n_states_seen:
        raise ValueError(f"No sHI data at all for panel {dataset}, fold {fold!r}.")
    n_states = n_states_seen.pop()

    missing = [i for i, c in enumerate(sHI_avg) if c is None]
    if missing:
        print(f"Warning: no sHI data for paths {missing} (panel {dataset}, fold {fold!r}) -- treating as 0 contribution.")
    sHI_avg = [c if c is not None else np.zeros(n_states) for c in sHI_avg]
    return sHI_avg, n_states


def animate_panel_AE(panel_number, n_pixels, state_to_show=None, fold="ensemble", sHI=None):
    dataset = str(panel_number)
    sHI_avg, n_states = _load_sHI_avg(panel_number, fold, sHI=sHI)

    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = (X, Y)

    U_arr = np.zeros_like(X)
    U(U_arr, grid)
    print(U_arr)

    # Pre-compute all maps to get a consistent colorbar scale across frames
    wcpdi_maps = []
    for state in range(n_states):
        P_arr = np.zeros_like(X)
        sHI_per_state = [curve[state] for curve in sHI_avg]
        
        print(f"State {state}: sHI_per_state = {sHI_per_state}")
        P_AE(P_arr, grid, sHI_per_state)
        print(f"State {state}: P_arr min={P_arr.min():.4f}, max={P_arr.max():.4f}")
        wcpdi_maps.append(WCPDI(P_arr, U_arr))

    # Global scale (fixed 0-1 across every frame) for the normalised panel.
    vmin_global, vmax_global = _safe_minmax(np.stack(wcpdi_maps))
    span_global = (vmax_global - vmin_global) or 1.0

    def _norm(m):
        return (m - vmin_global) / span_global

    fig, (ax, ax_norm) = plt.subplots(1, 2, figsize=(12, 8))
    _draw_static_panel(ax, panel_number)
    _draw_static_panel(ax_norm, panel_number)

    # Adaptive scale for the left panel: colours always span the current
    # frame's own [min, max], so the colorbar's numbers change every frame
    # while the colour mapping itself (hot, low->high) stays the same.
    m0 = wcpdi_maps[0]
    fmin0, fmax0 = _safe_minmax(m0)
    if fmax0 == fmin0:
        fmax0 = fmin0 + 1.0

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im0 = ax.imshow(m0.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP, vmin=fmin0, vmax=fmax0)
    cbar = fig.colorbar(im0, cax=cax, label='WCPDI Value')
    ax.set_title('Adaptive-scale WCPDI')

    divider_norm = make_axes_locatable(ax_norm)
    cax_norm = divider_norm.append_axes("right", size="5%", pad=0.05)
    im0_norm = ax_norm.imshow(_norm(m0).T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP, vmin=0, vmax=1)
    fig.colorbar(im0_norm, cax=cax_norm, label='Normalised WCPDI')
    ax_norm.set_title('Normalised WCPDI')

    title = ax.text(0.5, 1.05, f"State 0/{n_states - 1}",
                    transform=ax.transAxes, ha='center', va='bottom')

    # Snapshot of a single state, saved separately from the animation itself.
    if state_to_show is not None:
        WCPDI_map = wcpdi_maps[state_to_show]
        fmin, fmax = _safe_minmax(WCPDI_map)
        if fmax == fmin:
            fmax = fmin + 1.0

        fig_snap, (ax_s, ax_sn) = plt.subplots(1, 2, figsize=(12, 8))
        _draw_static_panel(ax_s, panel_number)
        _draw_static_panel(ax_sn, panel_number)

        im_s = ax_s.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP, vmin=fmin, vmax=fmax)
        div_s = make_axes_locatable(ax_s)
        fig_snap.colorbar(im_s, cax=div_s.append_axes("right", size="5%", pad=0.05), label='WCPDI Value')
        ax_s.set_title(f'WCPDI — State {state_to_show}')

        im_sn = ax_sn.imshow(_norm(WCPDI_map).T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP, vmin=0, vmax=1)
        div_sn = make_axes_locatable(ax_sn)
        fig_snap.colorbar(im_sn, cax=div_sn.append_axes("right", size="5%", pad=0.05), label='Normalised WCPDI')
        ax_sn.set_title(f'Normalised WCPDI — State {state_to_show}')

        for a in (ax_s, ax_sn):
            a.set_xlabel('X Position')
            a.set_ylabel('Y Position')
        fig_snap.tight_layout()
        fig_snap.savefig(f"WCPDI_AE_map_state_{state_to_show}.svg", format='svg', bbox_inches='tight')
        plt.show()

    def update(state):
        WCPDI_map = wcpdi_maps[state]
        print(WCPDI_map.T)

        fmin, fmax = _safe_minmax(WCPDI_map)
        if fmax == fmin:
            fmax = fmin + 1.0
        im0.set_data(WCPDI_map.T)
        im0.set_clim(fmin, fmax)
        cbar.update_normal(im0)

        im0_norm.set_data(_norm(WCPDI_map).T)

        title.set_text(f"State {state}/{n_states - 1}")
        return [im0, im0_norm, title]

    anim = animation.FuncAnimation(fig, update, frames=n_states, interval=200, blit=False)
    anim.save(f'panel_{panel_number}_WCPDI_AE_animation.gif', writer='pillow')


def plot_damage_map_grid(panel_numbers=None, fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
                          fold="ensemble", n_pixels=DEFAULT_N_PIXELS, sHI=None, save_path=None):
    """5x5 grid of raw (non-normalised) WCPDI damage maps: one row per panel, one
    column per lifetime fraction (state = round(fraction * (n_states - 1)) for that
    panel). Each subplot autoscales to its own data range -- values aren't
    comparable panel-to-panel or fraction-to-fraction, only the spatial pattern is."""
    if panel_numbers is None:
        panel_numbers = PANELS

    if sHI is None:
        cached = load_cached()
        if cached is None:
            raise FileNotFoundError(f"No cached results in {OUT_DIR} -- run path_performance.py first.")
        sHI, _ = cached

    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = (X, Y)

    U_arr = np.zeros_like(X)
    U(U_arr, grid)

    n_rows, n_cols = len(panel_numbers), len(fractions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for row, panel_number in enumerate(panel_numbers):
        sHI_avg, n_states = _load_sHI_avg(panel_number, fold, sHI=sHI)
        for col, fraction in enumerate(fractions):
            ax = axes[row][col]
            state = int(round(fraction * (n_states - 1)))

            P_arr = np.zeros_like(X)
            sHI_per_state = [curve[state] for curve in sHI_avg]
            P_AE(P_arr, grid, sHI_per_state)
            wcpdi_map = WCPDI(P_arr, U_arr)

            _draw_static_panel(ax, int(panel_number))
            im = ax.imshow(wcpdi_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='WCPDI Value')

            if row == 0:
                ax.set_title(f"{fraction:.0%} lifetime")
            if col == 0:
                ax.set_ylabel(f"Panel {panel_number}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    #fig.suptitle("WCPDI damage maps (AE) across lifetime")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()
    return fig


if __name__ == "__main__":
    
    panel_number = 123
    fold = "ensemble"

    cached = load_cached()
    if cached is None:
        raise FileNotFoundError(f"No cached results in {OUT_DIR} -- run path_performance.py first.")
    sHI, _ = cached

    x = np.linspace(0, PANEL_W, 100)
    y = np.linspace(0, PANEL_H, 100)
    grid = np.meshgrid(x, y, indexing='ij')

    sHI_avg, _ = _load_sHI_avg(panel_number, fold, sHI=sHI)
    sHI_per_state = [curve[0] for curve in sHI_avg]  # state 0, one sHI value per path

    P_values = np.zeros((len(x), len(y)))
    P_AE(P_values, grid, sHI_per_state)

    plt.figure(figsize=(6, 8))
    plt.imshow(P_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='Damage Probability')
    plt.title('Damage Probability Map (AE)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()

    U_values = np.zeros((len(x), len(y)))
    U(U_values, grid)

    plt.figure(figsize=(6, 8))
    plt.imshow(U_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='Weight Sum')
    plt.title('Weight Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()

    WCPDI_map = WCPDI(P_values, U_values)

    plt.figure(figsize=(6, 8))
    plt.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='WCPDI Value')
    plt.title('WCPDI Map (AE)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
    #plt.show()

    animate_panel_AE(panel_number=panel_number, n_pixels=DEFAULT_N_PIXELS, state_to_show=0, fold=fold, sHI=sHI)
    plot_damage_map_grid(panel_numbers=[103, 104, 105, 109, 123], fractions=(0.0, 0.25, 0.5, 0.75, 1.0),  n_pixels=DEFAULT_N_PIXELS, sHI=sHI, save_path=f"WCPDI_AE_damage_maps_grid.svg")