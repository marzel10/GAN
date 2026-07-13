'''
Visualizes the crossing graph defined by IS_CROSSING from weight_matrix.py.

Each node = one propagation path (1-28).
Node position = midpoint of its sensor segment → spatial layout on the panel.
Edges = pairs of paths that physically cross (IS_CROSSING[i,j] == True).

Run directly to show the plot.
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from weight_matrix import find_crossings, adjencency_matrix
from plot_panel import SENSOR_POSITIONS, SENSOR_PAIRS, PANEL_W, PANEL_H, DAMAGE_POINTS
from config import mat_file_path, DEFAULT_FREQ_INDEX, CMAP_SEQUENTIAL, CMAP_BLUES


def build_crossing_graph():
    is_crossing = find_crossings()
    G = nx.Graph()
    G.add_nodes_from(range(28))
    rows, cols = np.nonzero(np.triu(is_crossing))
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j))
    return G, is_crossing


def node_positions():
    """Midpoint of each sensor-pair segment — gives a spatial layout on the panel."""
    pos = {}
    for k, (s1, s2) in enumerate(SENSOR_PAIRS):
        p1 = SENSOR_POSITIONS[s1 - 1]
        p2 = SENSOR_POSITIONS[s2 - 1]
        pos[k] = (p1 + p2) / 2
    return pos


def visualize(show_path_lines=True):
    G, is_crossing = build_crossing_graph()
    pos = node_positions()                       # {node_idx: (x, y)}
    pos_for_nx = {k: (v[0], v[1]) for k, v in pos.items()}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- left: graph drawn on the panel ---
    ax = axes[0]
    rect = mpatches.Rectangle((0, 0), PANEL_W, PANEL_H,
                               linewidth=1.5, edgecolor="black",
                               facecolor="whitesmoke", zorder=0)
    ax.add_patch(rect)

    # sensor positions
    ax.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
               s=80, color="steelblue", zorder=4)
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, f"S{i}", ha="center", va="bottom",
                fontsize=8, color="steelblue", zorder=5)

    # faint path lines for context
    if show_path_lines:
        for s1, s2 in SENSOR_PAIRS:
            p1, p2 = SENSOR_POSITIONS[s1 - 1], SENSOR_POSITIONS[s2 - 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="lightgray", linewidth=0.8, zorder=1)

    # edges (crossing connections)
    for i, j in G.edges():
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot([xi, xj], [yi, yj], color="tomato", linewidth=1.0,
                alpha=0.6, zorder=2)

    # nodes (path midpoints)
    node_x = [pos[k][0] for k in range(28)]
    node_y = [pos[k][1] for k in range(28)]
    ax.scatter(node_x, node_y, s=60, color="darkorange",
               edgecolors="black", linewidths=0.5, zorder=3)
    for k in range(28):
        ax.text(pos[k][0], pos[k][1] + 0.005, str(k + 1),
                ha="center", va="bottom", fontsize=6, color="black", zorder=6)

    margin = 0.01
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Crossing graph on panel\n"
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
    )

    # legend
    ax.scatter([], [], s=60, color="darkorange", edgecolors="black",
               linewidths=0.5, label="Path node (midpoint)")
    ax.plot([], [], color="tomato", linewidth=1.0, label="Crossing edge")
    ax.plot([], [], color="lightgray", linewidth=0.8, label="Sensor path")
    ax.legend(loc="upper right", fontsize=8)

    # --- right: IS_CROSSING matrix as a heatmap ---
    ax2 = axes[1]
    im = ax2.imshow(is_crossing.astype(float), cmap=CMAP_BLUES, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax2, ticks=[0, 1], label="Crosses (1 = yes)", shrink=0.8)
    tick_labels = [str(i + 1) for i in range(28)]
    ax2.set_xticks(range(28))
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax2.set_yticks(range(28))
    ax2.set_yticklabels(tick_labels, fontsize=7)
    ax2.set_xlabel("Path index")
    ax2.set_ylabel("Path index")
    ax2.set_title("IS_CROSSING matrix")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.1f}")

    fig.tight_layout()
    plt.show()


def build_adjacency_graph(States, state_idx, freq_idx):
    """Weighted crossing graph for one state/frequency: edges only between
    crossing paths, weighted by adjencency_matrix (attention + crossing)."""
    adjacency = adjencency_matrix(States, state_idx, freq_idx)
    if adjacency.ndim != 2:
        raise ValueError(
            f"visualize_adjacency needs a single state (got state_idx={state_idx!r}, "
            f"which produced a {adjacency.ndim}D array). Pass one GLOBAL state index."
        )
    G = nx.Graph()
    G.add_nodes_from(range(28))
    rows, cols = np.nonzero(np.triu(adjacency))
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(adjacency[i, j]))
    return G, adjacency


def visualize_adjacency(States, state_idx, freq_idx, show_path_lines=True):
    """
    Visualizes the weighted adjacency graph from weight_matrix.adjencency_matrix:
      left  = graph drawn on the panel, edges colored/thickened by weight
      right = adjacency matrix heatmap

    state_idx : a single GLOBAL state index (int), not ":" — the adjacency
                matrix is only 2D (drawable as one graph) for a single state.
    """
    G, adjacency = build_adjacency_graph(States, state_idx, freq_idx)
    pos = node_positions()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- left: weighted graph drawn on the panel ---
    ax = axes[0]
    rect = mpatches.Rectangle((0, 0), PANEL_W, PANEL_H,
                               linewidth=1.5, edgecolor="black",
                               facecolor="whitesmoke", zorder=0)
    ax.add_patch(rect)

    # sensor positions
    ax.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
               s=80, color="steelblue", zorder=4)
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, f"S{i}", ha="center", va="bottom",
                fontsize=8, color="steelblue", zorder=5)

    # faint path lines for context
    if show_path_lines:
        for s1, s2 in SENSOR_PAIRS:
            p1, p2 = SENSOR_POSITIONS[s1 - 1], SENSOR_POSITIONS[s2 - 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="lightgray", linewidth=0.8, zorder=1)

    # weighted edges (color + thickness scale with adjacency weight)
    weights = [G[i][j]["weight"] for i, j in G.edges()]
    cmap = plt.get_cmap("plasma")
    wmin = min(weights) if weights else 0.0
    wmax = max(weights) if weights else 1.0
    norm = plt.Normalize(vmin=wmin, vmax=wmax if wmax > wmin else wmin + 1e-9)

    for i, j in G.edges():
        w = G[i][j]["weight"]
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot([xi, xj], [yi, yj], color=cmap(norm(w)),
                linewidth=0.8 + 2.5 * norm(w), alpha=0.8, zorder=2)

    # nodes (path midpoints)
    node_x = [pos[k][0] for k in range(28)]
    node_y = [pos[k][1] for k in range(28)]
    ax.scatter(node_x, node_y, s=60, color="darkorange",
               edgecolors="black", linewidths=0.5, zorder=3)
    for k in range(28):
        ax.text(pos[k][0], pos[k][1] + 0.005, str(k + 1),
                ha="center", va="bottom", fontsize=6, color="black", zorder=6)

    margin = 0.01
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Adjacency graph on panel (state {state_idx}, freq {freq_idx})\n"
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Edge weight (attention + crossing)", shrink=0.8)

    # legend
    ax.scatter([], [], s=60, color="darkorange", edgecolors="black",
               linewidths=0.5, label="Path node (midpoint)")
    ax.plot([], [], color="lightgray", linewidth=0.8, label="Sensor path")
    ax.legend(loc="upper right", fontsize=8)

    # --- right: adjacency matrix as a heatmap ---
    ax2 = axes[1]
    im = ax2.imshow(adjacency, cmap=CMAP_SEQUENTIAL)
    fig.colorbar(im, ax=ax2, label="Adjacency weight", shrink=0.8)
    tick_labels = [str(i + 1) for i in range(28)]
    ax2.set_xticks(range(28))
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax2.set_yticks(range(28))
    ax2.set_yticklabels(tick_labels, fontsize=7)
    ax2.set_xlabel("Path index")
    ax2.set_ylabel("Path index")
    ax2.set_title("Adjacency matrix")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    if G.number_of_edges():
        print(f"Weight range: {wmin:.4f} to {wmax:.4f}")

    fig.tight_layout()
    plt.show()


def _draw_path_panel(ax, path_number, neighbours, p1_t, p2_t):
    """Draw the panel view for a single path onto ax."""
    ax.add_patch(mpatches.Rectangle((0, 0), PANEL_W, PANEL_H,
                                    linewidth=1.5, edgecolor="black",
                                    facecolor="whitesmoke", zorder=0))
    for sa, sb in SENSOR_PAIRS:
        pa, pb = SENSOR_POSITIONS[sa - 1], SENSOR_POSITIONS[sb - 1]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color="lightgray", linewidth=0.8, zorder=1)

    for n in neighbours:
        sn1, sn2 = SENSOR_PAIRS[n]
        pn1, pn2 = SENSOR_POSITIONS[sn1 - 1], SENSOR_POSITIONS[sn2 - 1]
        ax.plot([pn1[0], pn2[0]], [pn1[1], pn2[1]],
                color="darkorange", linewidth=2.0, zorder=2)
        mx, my = (pn1 + pn2) / 2
        ax.text(mx, my + 0.005, str(n + 1),
                ha="center", va="bottom", fontsize=7, color="darkorange", zorder=5)

    ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
            color="steelblue", linewidth=2.5, zorder=3)
    mx_t, my_t = (p1_t + p2_t) / 2
    ax.text(mx_t, my_t + 0.005, str(path_number),
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color="steelblue", zorder=6)

    ax.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
               s=60, color="black", zorder=4)
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, f"S{i}", ha="center", va="bottom",
                fontsize=7, color="black", zorder=7)

    margin = 0.001
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    #ax.set_title("Panel view")
    ax.plot([], [], color="steelblue", linewidth=2.5, label=f"Path {path_number}")
    ax.plot([], [], color="darkorange", linewidth=2.0,
            label=f"Crossing paths ({len(neighbours)})")
    ax.plot([], [], color="lightgray", linewidth=0.8, label="Other paths")
    ax.legend(loc="upper right", fontsize=9)


def _draw_ego_graph(ax, G, idx, neighbours, path_number):
    """Draw the star ego-graph onto ax."""
    from matplotlib.lines import Line2D

    ego_nodes = [idx] + neighbours
    ego_G     = G.subgraph(ego_nodes)
    n_nb      = len(neighbours)
    angles    = np.linspace(0, 2 * np.pi, n_nb, endpoint=False)
    star_pos  = {idx: np.array([0.0, 0.0])}
    for nb, angle in zip(neighbours, angles):
        star_pos[nb] = np.array([np.cos(angle), np.sin(angle)])

    node_colors = ["steelblue" if n == idx else "darkorange" for n in ego_nodes]
    node_labels = {n: str(n + 1) for n in ego_nodes}

    nx.draw_networkx_edges(ego_G, star_pos, ax=ax,
                           edge_color="gray", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(ego_G, star_pos, ax=ax,
                           nodelist=ego_nodes, node_color=node_colors,
                           node_size=600, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(ego_G, star_pos, labels=node_labels, ax=ax,
                            font_size=9, font_color="white", font_weight="bold")

    ax.set_aspect("equal")
    ax.axis("off")
    #ax.set_title(f"Ego-graph for path {path_number}  ({n_nb} neighbours)")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=10, label=f"Path {path_number}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange",
               markersize=10, label="Crossing neighbour"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)


def visualize_path(path_number):
    """
    Renders three figures for a single path:
      1. Combined — panel view (left) + ego-graph (right) side by side.
      2. Panel view alone.
      3. Ego-graph alone.

    path_number : 1-indexed (1–28).
    """
    G, _ = build_crossing_graph()
    idx        = path_number - 1
    neighbours = sorted(G.neighbors(idx))
    s1_t, s2_t = SENSOR_PAIRS[idx]
    p1_t = SENSOR_POSITIONS[s1_t - 1]
    p2_t = SENSOR_POSITIONS[s2_t - 1]
    suptitle = f"Path {path_number}  (S{s1_t}–S{s2_t})  ·  {len(neighbours)} crossing paths"

    # ── 1. combined figure ────────────────────────────────────────────────
    fig, (ax_panel, ax_graph) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(suptitle, fontsize=12)
    _draw_path_panel(ax_panel, path_number, neighbours, p1_t, p2_t)
    _draw_ego_graph(ax_graph, G, idx, neighbours, path_number)
    fig.tight_layout()

    # ── 2. panel view alone ───────────────────────────────────────────────
    fig_panel, ax_p = plt.subplots(figsize=(7, 9))
    #fig_panel.suptitle(suptitle, fontsize=11)
    _draw_path_panel(ax_p, path_number, neighbours, p1_t, p2_t)
    fig_panel.tight_layout()

    # ── 3. ego-graph alone ────────────────────────────────────────────────
    fig_ego, ax_g = plt.subplots(figsize=(7, 7))
    #fig_ego.suptitle(suptitle, fontsize=11)
    _draw_ego_graph(ax_g, G, idx, neighbours, path_number)
    fig_ego.tight_layout()

    plt.show()


def plot_panel_schematic(impact_points=None, ax=None, title="Panel schematic",
                         show_dims=True, save_path=None):
    """
    Detailed panel schematic with sensors, stiffeners, disbond area and optional
    impact points — styled to sit alongside a MATLAB damage-map frame.

    impact_points : dict  {'label': np.array([x, y])}  positions in metres, or None.
    ax            : existing Axes to draw on, or None to create a new figure.
    show_dims     : draw span arrows with mm labels outside the panel (default True).
    save_path     : file path to save the figure (e.g. 'panel.svg').  None = no save.
    """
    # ── geometry constants ────────────────────────────────────────────────
    STIFFENERS = [(0.065, 0.004), (0.100, 0.004)]   # (x_centre, half_width) m
    X_GRID = [0.0, 0.025, 0.065, 0.100, 0.140, PANEL_W]
    Y_GRID = [0.0, 0.080, 0.160, PANEL_H]
    X_DIMS = [25, 40, 35, 40, 25]
    Y_DIMS = [80, 80, 80]
    SENSOR_R = 0.0085
    IMPACT_R = 0.0085

    # ── figure setup ──────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 5.5))
    else:
        fig = ax.get_figure()

    # Panel background
    ax.add_patch(mpatches.Rectangle(
        (0, 0), PANEL_W, PANEL_H,
        linewidth=1.5, edgecolor="black", facecolor="white", zorder=0,
    ))

    # Stiffeners (blue vertical bands)
    for x_c, hw in STIFFENERS:
        ax.add_patch(mpatches.Rectangle(
            (x_c - hw, 0), 2 * hw, PANEL_H,
            linewidth=0, facecolor="cornflowerblue", alpha=0.55, zorder=1,
        ))

    # Internal dashed grid lines
    for xg in X_GRID[1:-1]:
        ax.axvline(xg, color="dimgray", linestyle="--", linewidth=0.6, zorder=2)
    for yg in Y_GRID[1:-1]:
        ax.axhline(yg, color="dimgray", linestyle="--", linewidth=0.6, zorder=2)

    # Disbond area (panel 123)
    dp = DAMAGE_POINTS.get(123)
    if dp is not None and dp.ndim == 2:
        x_min, y_min = dp.min(axis=0)
        x_max, y_max = dp.max(axis=0)
        ax.add_patch(mpatches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1.5, edgecolor="darkred", facecolor="red",
            alpha=0.85, zorder=3,
        ))
        ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, "S23",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=4)

    # Sensors
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.add_patch(plt.Circle((x, y), SENSOR_R, color="dimgray", zorder=5))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=6)

    # Impact points
    if impact_points:
        for label, pos_m in impact_points.items():
            ax.add_patch(plt.Circle((pos_m[0], pos_m[1]), IMPACT_R,
                                    color="red", zorder=5))
            ax.text(pos_m[0], pos_m[1], label, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=6)

    # ── axes ──────────────────────────────────────────────────────────────
    # Small margin so the panel border isn't clipped; extra space only when
    # dimension arrows are drawn outside the panel.
    INNER  = 0.003                              # tiny gap inside axes limits
    OUTER  = 0.022 if show_dims else INNER      # room for arrows + labels

    ax.set_xlim(-INNER, PANEL_W + OUTER)
    ax.set_ylim(-OUTER, PANEL_H + INNER)
    ax.set_aspect("equal")
    ax.invert_xaxis()

    ax.set_xticks(X_GRID)
    ax.set_xticklabels([str(int(round(x * 1000))) for x in X_GRID], fontsize=7)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=2)
    ax.set_yticks(Y_GRID)
    ax.set_yticklabels([str(int(round(y * 1000))) for y in Y_GRID], fontsize=7)
    ax.set_ylabel("y (mm)", fontsize=8, labelpad=2)
    ax.tick_params(length=2, pad=2)

    # Dimension span annotations (optional)
    if show_dims:
        ann_kw = dict(arrowstyle="<->", color="black", lw=0.8)
        y_ann = -OUTER * 0.55

        for x0, x1, lbl in zip(X_GRID, X_GRID[1:], X_DIMS):
            ax.annotate("", xy=(x0, y_ann), xytext=(x1, y_ann),
                        arrowprops=ann_kw, annotation_clip=False)
            ax.text((x0 + x1) / 2, y_ann - 0.004, str(lbl),
                    ha="center", va="top", fontsize=6, clip_on=False)

        x_ann = PANEL_W + OUTER * 0.55
        for y0, y1, lbl in zip(Y_GRID, Y_GRID[1:], Y_DIMS):
            ax.annotate("", xy=(PANEL_W, y0), xytext=(PANEL_W, y1),
                        arrowprops=ann_kw, annotation_clip=False)
            ax.text(x_ann + 0.001, (y0 + y1) / 2, str(lbl),
                    ha="left", va="center", fontsize=6, clip_on=False)

    

    if standalone:
        fig.tight_layout(pad=0.4)
        if save_path is not None:
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


if __name__ == "__main__":
    visualize()
    visualize_path(2)

    # Example: weighted adjacency graph for one state/frequency of panel 123
    from states import states

    st_123_43 = states(str(mat_file_path("123_43")))
    visualize_adjacency(st_123_43, state_idx=70, freq_idx=DEFAULT_FREQ_INDEX)

    # Example: panel 123 with four impact cases
    impact_pts = {
        "C3": np.array([0.100, 0.100]),
        "C4": np.array([0.125, 0.100]),
        "C5": np.array([0.065, 0.160]),
        "C9": np.array([0.090, 0.150]),
    }
    plot_panel_schematic(impact_points=impact_pts, title="Panel 123",
                         save_path="panel_123_schematic.svg")
