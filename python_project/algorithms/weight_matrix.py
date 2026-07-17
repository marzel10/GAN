'''
This file:
- defines the attention_matrix function to compute the attention values based on signal energy for each path pair
- defines the find_crossings function to determine which paths cross each other based on sensor positions 
  (if two paths share a sensor, they are not crossing; if they connect different pairs of sensors that intersect in space, they are crossing)
- defines the adjencency_matrix function to create an adjacency matrix for the graph dataset, where edges exist only between crossing paths and are weighted by the attention values
- includes a main block that demonstrates how to compute and visualize the attention and adjacency matrices for a
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

from states import states
import numpy as np
import matplotlib.pyplot as plt
from plot_panel import SENSOR_POSITIONS, SENSOR_PAIRS
from config import NR_SAVED_STATES, STATE_START_INDICES, FAILURES_RECORD, mat_file_path, DEFAULT_FREQ_INDEX, CMAP_SEQUENTIAL, CMAP_BLUES

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
        
            if np.any(energy_i == 0) or np.any(energy_j == 0):
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

    is_crossing = np.zeros((28, 28), dtype=float)
    for i in range(28):
        sensors_i = SENSOR_PAIRS[i]
        p1_i, p2_i = SENSOR_POSITIONS[sensors_i[0] - 1], SENSOR_POSITIONS[sensors_i[1] - 1]
        path_i_vector = p2_i - p1_i
        
        for j in range(i + 1, 28):
            sensors_j = SENSOR_PAIRS[j]
            p1_j, p2_j = SENSOR_POSITIONS[sensors_j[0] - 1], SENSOR_POSITIONS[sensors_j[1] - 1]
            path_j_vector = p2_j - p1_j
            cos_theta = np.dot(path_i_vector, path_j_vector) / (np.linalg.norm(path_i_vector) * np.linalg.norm(path_j_vector)) # angle between the two paths
            cos_theta = max(0.0, cos_theta)
            if sensors_i[0] in sensors_j or sensors_i[1] in sensors_j: # Paths share a sensor
                shared_sensor = sensors_i[0] if sensors_i[0] in sensors_j else sensors_i[1]
                other_i = sensors_i[1] if shared_sensor == sensors_i[0] else sensors_i[0]
                other_j = sensors_j[1] if shared_sensor == sensors_j[0] else sensors_j[0]
                shared_pos = SENSOR_POSITIONS[shared_sensor - 1]
                vec_i_away = SENSOR_POSITIONS[other_i - 1] - shared_pos
                vec_j_away = SENSOR_POSITIONS[other_j - 1] - shared_pos
                cos_theta_shared = np.dot(vec_i_away, vec_j_away) / (np.linalg.norm(vec_i_away) * np.linalg.norm(vec_j_away))
                cos_theta_shared = max(0.0, cos_theta_shared)

                is_crossing[i, j] = cos_theta_shared
                is_crossing[j, i] = cos_theta_shared
                continue
            denom = (p2_i[0] - p1_i[0]) * (p2_j[1] - p1_j[1]) - (p2_i[1] - p1_i[1]) * (p2_j[0] - p1_j[0])
            if denom == 0:
                continue  
            ua = ((p2_j[0] - p1_j[0]) * (p1_i[1] - p1_j[1]) - (p2_j[1] - p1_j[1]) * (p1_i[0] - p1_j[0])) / denom
            ub = ((p2_i[0] - p1_i[0]) * (p1_i[1] - p1_j[1]) - (p2_i[1] - p1_i[1]) * (p1_i[0] - p1_j[0])) / denom
            eps = 0
            if eps < ua < 1 - eps and eps < ub < 1 - eps:
                is_crossing[i, j] = cos_theta + 1
                is_crossing[j, i] = cos_theta + 1

    return is_crossing

IS_CROSSING = find_crossings()

# FAILURES_RECORD is owned by config.py; imported above.

def _global_failure_threshold(panel_key):
    """Resolve the failure recorded for this panel as a GLOBAL state
    threshold: (global_failed_state, failed_sensor), or None if no failure
    is recorded for it.

    For panel 123, FAILURES_RECORD is keyed by the subpanel the failure was
    first observed in (e.g. "123_42"), but a physically broken sensor stays
    broken for every later subpanel too -- so the recorded local state is
    converted to a global one here, and the threshold applies to any
    "123_*" panel_key, not just the one it was recorded against.
    """
    panel_key = str(panel_key)
    if panel_key.startswith("123"):
        for key, (local_state, sensor) in FAILURES_RECORD.items():
            if key.startswith("123"):
                start_idx = STATE_START_INDICES.get(key, 0)
                return start_idx + local_state, sensor
        return None
    failure_info = FAILURES_RECORD.get(panel_key)
    if failure_info is None:
        return None
    return tuple(failure_info)

def failed_sensor_at(panel_number, state):
    """Return the sensor id (1-8) that has failed by GLOBAL `state` for this
    panel, or None if no failure applies yet (or ever). `state` must be the
    global state index, consistent with how Panel_GraphDataset /
    features_GraphDataset label states (including for panel 123)."""
    threshold = _global_failure_threshold(panel_number)
    if threshold is None:
        return None
    failed_state, failed_sensor = threshold
    return failed_sensor if state >= failed_state else None

def _paths_touching_sensor(sensor):
    """Path indices (0-27) whose SENSOR_PAIRS entry includes this sensor id (1-8)."""
    return {k for k, (a, b) in enumerate(SENSOR_PAIRS) if a == sensor or b == sensor}

def adjencency_matrix(States, state_idx: (int or str), freq_idx: int) -> np.ndarray:

    failure_threshold = _global_failure_threshold(States.panel_name)
    failed_paths = _paths_touching_sensor(failure_threshold[1]) if failure_threshold is not None else None

    attention = attention_matrix(States, state_idx, freq_idx)
    if (attention == 0).all():
        print("Warning: Attention matrix is all zeros. Check if the attention values are correct.")

    subpanel_start = STATE_START_INDICES.get(States.panel_name, 0) if States.panel_name.startswith("123") else 0

    if state_idx == ":":
        adjacency = np.zeros((States.num_states, 28, 28), dtype=float)
        for s in range(States.num_states):
            global_s = s + subpanel_start
            for i in range(28):
                for j in range(i + 1, 28):
                    if IS_CROSSING[i, j]>0:

                        if failed_paths is not None and global_s >= failure_threshold[0] and (i in failed_paths or j in failed_paths):
                            # Skip paths through the failed sensor for the affected states
                            continue
                        adjacency[s, i, j] = attention[s, i, j] + IS_CROSSING[i, j]
                        adjacency[s, j, i] = attention[s, i, j] + IS_CROSSING[i, j]

    else:
        # state_idx is GLOBAL by convention (attention_matrix shifts it to
        # local internally when indexing this States object's own arrays)
        adjacency = np.zeros((28, 28), dtype=float)
        for i in range(28):
            for j in range(i + 1, 28):
                eps = 10e-6
                if IS_CROSSING[i, j] > eps:
                    if failed_paths is not None and state_idx >= failure_threshold[0] and (i in failed_paths or j in failed_paths):
                        # Skip paths through the failed sensor for the affected state
                        continue
                    adjacency[i, j] = attention[i, j] + IS_CROSSING[i, j]
                    adjacency[j, i] = attention[i, j] + IS_CROSSING[i, j]

    return adjacency


if __name__ == "__main__":

    st_103 = states(str(mat_file_path("103")))
    state_i = ":"
    freq_i = DEFAULT_FREQ_INDEX
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
            im = ax.imshow(mat, cmap=CMAP_SEQUENTIAL, vmin=vmin, vmax=vmax)
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
        im = ax.imshow(attention_103, cmap=CMAP_SEQUENTIAL)
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

    im_c = ax_c.imshow(crossing, cmap=CMAP_BLUES)
    fig2.colorbar(im_c, ax=ax_c, label="Crosses (0/1)", ticks=[0, 1], shrink=0.8)
    ax_c.set_title("Path Crossing Matrix")
    ax_c.set_xlabel("Path Index")
    ax_c.set_ylabel("Path Index")
    ax_c.set_xticks(range(st_103.num_pair))
    ax_c.set_xticklabels(tick_labels, rotation=90)
    ax_c.set_yticks(range(st_103.num_pair))
    ax_c.set_yticklabels(tick_labels)

    im_a = ax_a.imshow(adj, cmap=CMAP_SEQUENTIAL)
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


    