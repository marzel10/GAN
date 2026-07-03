'''
This file:
- defines the attention_matrix function to compute the attention values based on signal energy for each path pair
- defines the find_crossings function to determine which paths cross each other based on sensor positions 
  (if two paths share a sensor, they are not crossing; if they connect different pairs of sensors that intersect in space, they are crossing)
- defines the adjencency_matrix function to create an adjacency matrix for the graph dataset, where edges exist only between crossing paths and are weighted by the attention values
- includes a main block that demonstrates how to compute and visualize the attention and adjacency matrices for a
'''
from states import states
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_panel import SENSOR_POSITIONS, SENSOR_PAIRS


NR_SAVED_STATES ={"123_1": 16, "123_2": 7, "123_31": 16, "123_32": 13, "123_41": 10, "123_42": 10, "123_43": 10, "123_44": 10}
STATE_START_INDICES = {"123_1": 0, "123_2": 16, "123_31": 23, "123_32": 39, "123_41": 52, "123_42": 62, "123_43": 72, "123_44": 82}

def attention_matrix(States: states, state_idx: (int or str), freq_idx: int) -> np.ndarray:
    num_paths = States.num_pair
    num_states = States.num_states

    # shift state_idx if it's a string and panel is 123
    if States.panel_name.startswith("123") and state_idx != ":":

        # find in state_start_idx based on the key
        start_idx =STATE_START_INDICES.get(States.panel_name, None)
        state_idx -= start_idx if start_idx is not None else 0


    if isinstance(state_idx, str):
        if state_idx == ":":
            attention = np.zeros((num_states, num_paths, num_paths))
        else:
            raise ValueError(f"Invalid state index string: {state_idx}")
    else:
        attention = np.zeros((num_paths, num_paths))

    for i in range(num_paths):
        amp_i = States.amplitude(state_idx, freq_idx, i)
        
        energy_i =States.signal_energy(state_idx, freq_idx, i)
        for j in range(num_paths):
            amp_j = States.amplitude(state_idx, freq_idx, j)
            energy_j = States.signal_energy(state_idx, freq_idx, j)
            if energy_i == 0 or energy_j == 0:
                print(f"Warning: Zero energy for state {state_idx}, frequency {freq_idx}, paths {i} or {j}. Check if the amplitude values are correct.")
                
            
            if isinstance(state_idx, str) and state_idx == ":":
                attention[:, i, j] = energy_i + energy_j
            else:
                attention[i, j] = energy_i + energy_j
    if attention.all() == 0:    
        print(f"WARNING: All attention values are zero for state {state_idx}. Check if the energy calculations are correct.")
        # Optional: Normalize the attention values
    if isinstance(state_idx, str) and state_idx == ":":

        attention = attention - np.min(attention, axis=(1, 2), keepdims=True)
        attention /= np.max(attention, axis=(1, 2), keepdims=True) - np.min(attention, axis=(1, 2), keepdims=True) 
    else:
        attention = attention - np.min(attention)
        attention /= np.max(attention) - np.min(attention) 

    if (attention == 0).all():
        print("Warning: All attention values are zero. Check if the energy calculations are correct.")

    return attention

def find_crossings():

    is_crossing = np.zeros((28, 28), dtype=bool)
    for i in range(28):
        p1_i, p2_i = SENSOR_POSITIONS[SENSOR_PAIRS[i][0] - 1], SENSOR_POSITIONS[SENSOR_PAIRS[i][1] - 1]
        for j in range(i + 1, 28):
            p1_j, p2_j = SENSOR_POSITIONS[SENSOR_PAIRS[j][0] - 1], SENSOR_POSITIONS[SENSOR_PAIRS[j][1] - 1]
            denom = (p2_i[0] - p1_i[0]) * (p2_j[1] - p1_j[1]) - (p2_i[1] - p1_i[1]) * (p2_j[0] - p1_j[0])
            if denom == 0:
                continue
            ua = ((p2_j[0] - p1_j[0]) * (p1_i[1] - p1_j[1]) - (p2_j[1] - p1_j[1]) * (p1_i[0] - p1_j[0])) / denom
            ub = ((p2_i[0] - p1_i[0]) * (p1_i[1] - p1_j[1]) - (p2_i[1] - p1_i[1]) * (p1_i[0] - p1_j[0])) / denom
            eps = 1e-9
            if eps < ua < 1 - eps and eps < ub < 1 - eps:
                is_crossing[i, j] = True
                is_crossing[j, i] = True

    return is_crossing

IS_CROSSING = find_crossings()

FAILURES_RECORD ={
    # panel_name: [relative_state_idex, sensor_idex]
    "123_42": [9,3],
    "104": [15, 8],
    "105": [16, 8]
}

def adjencency_matrix(States, state_idx: (int or str), freq_idx: int) -> np.ndarray:

    failure_info = FAILURES_RECORD.get(States.panel_name, None)
    if failure_info is not None:
        failure_state_idx, failure_sensor_idx = failure_info
    
    attention = attention_matrix(States, state_idx, freq_idx)
    if (attention == 0).all():
        print("Warning: Attention matrix is all zeros. Check if the attention values are correct.")
    if state_idx == ":":
        adjacency = np.zeros((States.num_states, 28, 28), dtype=float)
        for s in range(States.num_states):
            for i in range(28):
                for j in range(i + 1, 28):
                    if IS_CROSSING[i, j]:

                        if failure_info is not None and s >= failure_state_idx and (i == failure_sensor_idx or j == failure_sensor_idx):
                            # Skip the failed sensor for the affected states
                            continue
                        adjacency[s, i, j] = attention[s, i, j]
                        adjacency[s, j, i] = attention[s, i, j]

    else:
        # match the state-index frame FAILURES_RECORD uses (relative to subpanel start for "123_*" panels)
        local_state_idx = state_idx
        if failure_info is not None and States.panel_name.startswith("123"):
            start_idx = STATE_START_INDICES.get(States.panel_name, None)
            local_state_idx = state_idx - start_idx if start_idx is not None else state_idx

        adjacency = np.zeros((28, 28), dtype=float)
        for i in range(28):
            for j in range(i + 1, 28):
                if IS_CROSSING[i, j]:
                    if failure_info is not None and local_state_idx >= failure_state_idx and (i == failure_sensor_idx or j == failure_sensor_idx):
                        # Skip the failed sensor for the affected state
                        continue
                    adjacency[i, j] = attention[i, j]
                    adjacency[j, i] = attention[i, j]

    return adjacency


if __name__ == "__main__":

    _DATA_DIR = Path(__file__).resolve().parent

    st_103 = states(str(_DATA_DIR / "data/States_103.mat"))
    state_i = ":"
    freq_i = 3
    attention_103 = attention_matrix(st_103, state_i, freq_i)
    if attention_103.ndim == 3:
        random_state_indices = np.random.choice(st_103.num_states, size=3, replace=False)
        matrices = [attention_matrix(st_103, idx, freq_i) for idx in random_state_indices]
        vmin = min(m.min() for m in matrices)
        print(vmin)
        vmax = max(m.max() for m in matrices)
        print(vmax)
        print(vmax - vmin)
        print(f"Attention value range across states: {vmin:.4f} to {vmax:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(18,6))
        tick_labels = [f" {i+1}" for i in range(st_103.num_pair)]
        for ax, mat, idx in zip(axes, matrices, random_state_indices):
            im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"State {idx}")
            ax.set_xticks(range(st_103.num_pair))
            ax.set_xticklabels(tick_labels, rotation=90)
            ax.set_yticks(range(st_103.num_pair))
            ax.set_yticklabels(tick_labels)
        fig.colorbar(im, ax=axes, label="Attention Value", shrink=0.8,
                     ticks=np.linspace(vmin, vmax, 6))
        fig.suptitle("Attention Matrices for Panel 103")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attention_103, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Attention Value",
                     ticks=np.linspace(0, 1, 6))
        ax.set_title("Attention Matrix for Panel 103")
        tick_labels = [f"{i+1}" for i in range(st_103.num_pair)]
        ax.set_xticks(range(st_103.num_pair))
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.set_yticks(range(st_103.num_pair))
        ax.set_yticklabels(tick_labels)
        fig.tight_layout()

    # --- is_crossing and adjacency matrices ---
    tick_labels = [f"{i+1}" for i in range(st_103.num_pair)]
    sample_idx = random_state_indices[0] if attention_103.ndim == 3 else state_i

    crossing = find_crossings().astype(float)
    adj = adjencency_matrix(st_103, sample_idx, freq_i)
    print(adj)

    fig2, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(14, 6))

    im_c = ax_c.imshow(crossing, cmap="Blues", vmin=0, vmax=1)
    fig2.colorbar(im_c, ax=ax_c, label="Crosses (0/1)", ticks=[0, 1], shrink=0.8)
    ax_c.set_title("Path Crossing Matrix")
    ax_c.set_xlabel("Path Index")
    ax_c.set_ylabel("Path Index")
    ax_c.set_xticks(range(st_103.num_pair))
    ax_c.set_xticklabels(tick_labels, rotation=90)
    ax_c.set_yticks(range(st_103.num_pair))
    ax_c.set_yticklabels(tick_labels)

    im_a = ax_a.imshow(adj, cmap="viridis")
    fig2.colorbar(im_a, ax=ax_a, label="Attention (crossing paths only)",
                  ticks=np.linspace(adj.min(), adj.max(), 6), shrink=0.8)
    ax_a.set_title(f"Adjacency Matrix (state {sample_idx})")
    ax_a.set_xlabel("Path Index")
    ax_a.set_ylabel("Path Index")
    ax_a.set_xticks(range(st_103.num_pair))
    ax_a.set_xticklabels(tick_labels, rotation=90)
    ax_a.set_yticks(range(st_103.num_pair))
    ax_a.set_yticklabels(tick_labels)

    fig2.tight_layout()
    plt.show()