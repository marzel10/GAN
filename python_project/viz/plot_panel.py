'''
This file defines functions to visualize the PZT panel, including:
- plot_panel_with_paths: draw the panel with sensors, damage point, and optionally active paths
- animate_panel_sHI: create an animation of the panel with path colours representing sHI values across states 
colors are only on the paths so there is no proper heatmap of the panel, but it gives a visual representation of how the sHI values change across states and paths.

The panel geometry and sensor positions are defined as constants at the top of the file, matching the experimental setup.

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from itertools import combinations

# Panel geometry (metres)
PANEL_W = 0.165   # 165 mm
PANEL_H = 0.240   # 240 mm

# Sensor positions in metres (1-indexed, matches MATLAB SP_table order)
SENSOR_POSITIONS = np.array([
    [0.025,  0.025],   # S1
    [0.100,  0.025],   # S2
    [0.140,  0.215],   # S3
    [0.065,  0.215],   # S4
    [0.025,  0.120],   # S5
    [0.140,  0.120],   # S6
    [0.0825, 0.080],   # S7
    [0.0825, 0.160],   # S8
])

# All 28 unique sensor-pair paths, path index k → SENSOR_PAIRS[k-1]
SENSOR_PAIRS = [(i, j) for i, j in combinations(range(1, 9), 2)]  # 1-indexed

# Known impact positions (metres) per panel
DAMAGE_POINTS = {
    103: np.array([0.050,  0.080]),
    104: np.array([0.025,  0.080]),
    105: np.array([0.115,  0.160]),   # (165-50, 240-80) mm
    109: np.array([0.0825, 0.140]),   # (165-82.5, 240-100) mm
    123: np.array([[0.0825-0.03,0.045],[0.0825,0.045],[0.0825-0.03,0.045+0.03],[0.0825,0.045+0.03]]),   # square debond surface
}



def _draw_static_panel(ax, panel_number):
    """Draw panel boundary, sensors, and damage point onto ax."""
    rect = mpatches.Rectangle(
        (0, 0), PANEL_W, PANEL_H,
        linewidth=1.5, edgecolor="black", facecolor="whitesmoke", zorder=0,
    )
    ax.add_patch(rect)

    ax.scatter(
        SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
        s=60, color="steelblue", zorder=3,
    )
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, str(i), ha="center", va="bottom",
                fontsize=8, color="steelblue", zorder=4)

    dp = DAMAGE_POINTS.get(panel_number)
    if dp is not None and dp.ndim == 1:
        ax.scatter(dp[0], dp[1], s=120, color="red", marker="X", zorder=5)
        ax.text(dp[0], dp[1] + 0.008, "Impact",
                ha="center", va="bottom", fontsize=8, color="red", zorder=6)
    elif dp is not None and dp.ndim == 2:
        x_min, y_min = dp.min(axis=0)
        x_max, y_max = dp.max(axis=0)
        width, height = x_max - x_min, y_max - y_min
        rect = mpatches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--", zorder=5,
            label="Debond area",
        )
        ax.add_patch(rect)
    margin = 0.01
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def animate_panel_sHI(panel_number, path_indices, sHI_matrix,
                      interval=150, cmap="plasma", save_path=None):
    """
    Animate sHI values on the panel paths over states.

    Each path line is coloured by its sHI value at the current state;
    colours are normalised globally so they are comparable across frames.

    Parameters
    ----------
    panel_number : int
        103, 104, 105, or 109.
    path_indices : list[int]
        1-indexed path numbers (1-28), length n_paths.
    sHI_matrix : ndarray of shape (n_paths, n_states)
        Scalar sHI per path per state (e.g. mean of latent dims).
    interval : int
        Milliseconds between frames.
    cmap : str
        Matplotlib colormap name.  ``"plasma"`` works well; use
        ``"RdYlGn"`` if high sHI means healthy and low means damaged.
    save_path : str | None
        If given, save the animation to this path (.gif or .mp4).

    Returns
    -------
    anim : FuncAnimation
    fig  : matplotlib Figure
    """
    n_paths, n_states = sHI_matrix.shape
    assert len(path_indices) == n_paths, (
        f"path_indices length {len(path_indices)} != sHI_matrix rows {n_paths}"
    )

    norm = mcolors.Normalize(vmin=sHI_matrix.min(), vmax=sHI_matrix.max())
    colormap = plt.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(5, 7))
    plt.subplots_adjust(right=0.78)   # room for colourbar

    _draw_static_panel(ax, panel_number)

    # Create one Line2D per path, coloured by sHI at state 0
    lines = []
    for k, path_idx in enumerate(path_indices):
        if not (1 <= path_idx <= 28):
            lines.append(None)
            continue
        s1, s2 = SENSOR_PAIRS[path_idx - 1]
        p1, p2 = SENSOR_POSITIONS[s1 - 1], SENSOR_POSITIONS[s2 - 1]
        colour = colormap(norm(sHI_matrix[k, 0]))
        (line,) = ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=colour, linewidth=2.0, alpha=0.9, zorder=2,
        )
        lines.append(line)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="sHI", shrink=0.6)

    title_obj = ax.set_title(f"Panel {panel_number} — state 1 / {n_states}")

    def update(frame):
        for k, line in enumerate(lines):
            if line is None:
                continue
            colour = colormap(norm(sHI_matrix[k, frame]))
            line.set_color(colour)
        title_obj.set_text(f"Panel {panel_number} — state {frame + 1} / {n_states}")
        return [l for l in lines if l is not None] + [title_obj]

    anim = FuncAnimation(
        fig, update, frames=n_states, interval=interval, blit=True,
    )

    if save_path:
        anim.save(save_path)

    return anim, fig
