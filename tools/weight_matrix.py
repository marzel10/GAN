'''
This file contains the following functions:
- attention_matrix: Computes connection strength based only on raw signal features
- find_crossings: Determines which paths cross each other based on  angles and crossings principles
- find_crossings_by_area: Computes the crossing strength between paths based on the area of overlap of their coverage regions
- failed_sensor_at: Returns the sensor id (1-8) that has failed by GLOBAL `state` for this panel, or None if no failure applies
- adjacency_matrix: Constructs adjacency matrix for different types of connections and excludes broken sensors
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

from states import states
import numpy as np
import matplotlib.pyplot as plt
from config import BETA_CONSTANT, STATE_START_INDICES, FAILURES_RECORD, mat_file_path, DEFAULT_FREQ_INDEX, CMAP_SEQUENTIAL, CMAP_BLUES, SENSOR_POSITIONS, SENSOR_PAIRS

def attention_matrix(States: states, state_idx: (int or str), freq_idx: int, energy: bool=True) -> np.ndarray:
    '''Computes connecion strength based only on raw signal features'''
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
        
        if energy:
            strength_i =States.signal_energy(state_idx, freq_idx, i)
        else:
            # compute strength_i using the geometric mean of the signal envelope peak and area
            strength_i = np.sqrt(States.signal_envelope_peak(state_idx, freq_idx, i)*States.signal_envelope_peak_area(state_idx, freq_idx, i))

        for j in range(num_paths):
            
            if energy:
                strength_j = States.signal_energy(state_idx, freq_idx, j)
            else:
                strength_j = np.sqrt(States.signal_envelope_peak(state_idx, freq_idx, j)*States.signal_envelope_peak_area(state_idx, freq_idx, j))
        
            if np.any(strength_i == 0) or np.any(strength_j == 0):
                print(f"Warning: Zero energy for state {state_idx}, frequency {freq_idx}, paths {i} or {j}. Check if the amplitude values are correct.")  
            
            if isinstance(state_idx, str) and state_idx == ":":
                attention[:, i, j] = strength_i + strength_j
            else:
                attention[i, j] = strength_i + strength_j
    if attention.all() == 0:    
        print(f"WARNING: All attention values are zero for state {state_idx}. Check if the energy calculations are correct.")
    
    return attention



def find_crossings():
    '''This function derives geometric strength of connections following angles and crossings principles'''
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
            if denom == 0: # Paths are parallel or collinear
                continue  
            ua = ((p2_j[0] - p1_j[0]) * (p1_i[1] - p1_j[1]) - (p2_j[1] - p1_j[1]) * (p1_i[0] - p1_j[0])) / denom
            ub = ((p2_i[0] - p1_i[0]) * (p1_i[1] - p1_j[1]) - (p2_i[1] - p1_i[1]) * (p1_i[0] - p1_j[0])) / denom
            eps = 0
            if eps < ua < 1 - eps and eps < ub < 1 - eps:
                is_crossing[i, j] = abs(cos_theta) + 1
                is_crossing[j, i] = abs(cos_theta) + 1

    # Normalize the crossing values to be between 0 and 1
    is_crossing = is_crossing - np.min(is_crossing)
    is_crossing /= np.max(is_crossing) - np.min(is_crossing)

    return is_crossing


def find_crossings_by_area(n_pixels=None, panel_number=None, state=None, beta_constant=None):
    ''' This function computes the crossing strength between paths based on the area of overlap of their coverage regions.'''

    from imagining_alghoritm import U
    from config import PANEL_W, PANEL_H, DEFAULT_N_PIXELS

    if n_pixels is None:
        n_pixels = DEFAULT_N_PIXELS
    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    grid = np.meshgrid(x, y, indexing="ij")
    X = grid[0]

    n = len(SENSOR_PAIRS)
    
    coverage_masks = []
    for idx in range(n):
        U_arr = np.zeros_like(X)
        U(U_arr, grid, panel_number=panel_number, state=state, path_indices=[idx], beta_constant=beta_constant)
        coverage_masks.append(U_arr > 0)

    areas = np.array([mask.sum() for mask in coverage_masks], dtype=float)

    strength = np.zeros((n, n))
    for i in range(n):
        if areas[i] == 0:
            continue
        for j in range(n):
            common = np.sum(coverage_masks[i] & coverage_masks[j])
            strength[i, j] = common / areas[i]

    return strength


def _global_failure_threshold(panel_key):
    """Resolve the failure recorded for this panel as a GLOBAL state
    threshold: (global_failed_state, failed_sensor), or None if no failure
    is recorded for it."""
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
    panel, or None if no failure applies """
    threshold = _global_failure_threshold(panel_number)
    if threshold is None:
        return None
    failed_state, failed_sensor = threshold
    return failed_sensor if state >= failed_state else None

def _paths_touching_sensor(sensor):
    """Path indices (0-27) whose SENSOR_PAIRS entry includes this sensor id (1-8)."""
    return {k for k, (a, b) in enumerate(SENSOR_PAIRS) if a == sensor or b == sensor}

def adjencency_matrix(States, state_idx: (int or str), freq_idx: int, type: str="basic", beta_constant: float=BETA_CONSTANT) -> np.ndarray:
    '''This function construct adjacency matrix for different types of connections and excludes broken sensors'''
    failure_threshold = _global_failure_threshold(States.panel_name)
    failed_paths = _paths_touching_sensor(failure_threshold[1]) if failure_threshold is not None else None

    if type == "peak":
        attention = attention_matrix(States, state_idx, freq_idx, energy=False)
    else:
        attention = attention_matrix(States, state_idx, freq_idx)
    if (attention == 0).all():
        print("Warning: Attention matrix is all zeros. Check if the attention values are correct.")

    subpanel_start = STATE_START_INDICES.get(States.panel_name, 0) if States.panel_name.startswith("123") else 0

    if state_idx == ":":
        adjacency = np.zeros((States.num_states, 28, 28), dtype=float)
        for s in range(States.num_states):
            global_s = s + subpanel_start

            if type == "by_area":
                crossing_strength = find_crossings_by_area(panel_number=States.panel_name, state=global_s, beta_constant=beta_constant)
            else:
                crossing_strength = find_crossings()

            for i in range(28):
                for j in range(i + 1, 28):
                    if crossing_strength[i, j] > 0:

                        if failed_paths is not None and global_s >= failure_threshold[0] and (i in failed_paths or j in failed_paths):
                            # Skip paths through the failed sensor for the affected states
                            continue

                        if type == "basic" or type == "peak" or type == "by_area":
                            adjacency[s, i, j] = attention[s, i, j] * crossing_strength[i, j]
                            adjacency[s, j, i] = attention[s, i, j] * crossing_strength[j, i]
                        elif type == "geometry":
                            adjacency[s, i, j] =  crossing_strength[i, j] 
                            adjacency[s, j, i] =  crossing_strength[j, i] 
                        else:
                            raise ValueError(f"Unknown adjacency matrix type: {type}")
    else:
        
        adjacency = np.zeros((28, 28), dtype=float)
        if type == "by_area":
            crossing_strength = find_crossings_by_area(panel_number=States.panel_name, state=state_idx, beta_constant=beta_constant)
        else:
            crossing_strength = find_crossings()

        for i in range(28):
            for j in range(i + 1, 28):
                eps = 10e-6
                if crossing_strength[i, j] > eps:
                    if failed_paths is not None and state_idx >= failure_threshold[0] and (i in failed_paths or j in failed_paths):
                        # Skip paths through the failed sensor for the affected state
                        continue

                    if type == "basic" or type == "peak" or type == "by_area":
                        adjacency[i, j] = attention[i, j] * crossing_strength[i, j]
                        adjacency[j, i] = attention[i, j] * crossing_strength[j, i]
                    elif type == "geometry":
                        adjacency[i, j] =  crossing_strength[i, j] 
                        adjacency[j, i] =  crossing_strength[j, i]
                    else:
                        raise ValueError(f"Unknown adjacency matrix type: {type}")

    return adjacency


def _plot_path_matrix(fig, ax, mat, title, cmap, cbar_label, ticks=None):
    """Shared imshow-with-path-index-ticks helper for the showcases below."""
    im = ax.imshow(mat, cmap=cmap,
                    vmin=ticks[0] if ticks is not None else None,
                    vmax=ticks[-1] if ticks is not None else None)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8, ticks=ticks)
    ax.set_title(title)
    ax.set_xlabel("Path index")
    ax.set_ylabel("Path index")
    n = mat.shape[0]
    tick_labels = [str(i + 1) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=7)
    return im


if __name__ == "__main__":

    st_103 = states(str(mat_file_path("103")))
    freq_i = DEFAULT_FREQ_INDEX

    # --- 1. attention_matrix: energy-based connection strength, 3 random states ---
    random_states = np.random.choice(st_103.num_states, size=3, replace=False)
    attention_mats = [attention_matrix(st_103, int(idx), freq_i) for idx in random_states]
    vmin = min(m.min() for m in attention_mats)
    vmax = max(m.max() for m in attention_mats)
    print(f"Attention value range across states: {vmin:.4f} to {vmax:.4f}")

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    for ax, mat, idx in zip(axes1, attention_mats, random_states):
        _plot_path_matrix(fig1, ax, mat, f"State {idx}", CMAP_SEQUENTIAL,
                           "Attention value", ticks=np.linspace(vmin, vmax, 6))
    fig1.suptitle("attention_matrix -- Panel 103, 3 sample states")
    fig1.tight_layout()

    # --- 2. find_crossings vs find_crossings_by_area: two ways to score path crossings ---
    fig2, (ax_c, ax_ca) = plt.subplots(1, 2, figsize=(14, 6))
    _plot_path_matrix(fig2, ax_c, find_crossings(), "find_crossings (angle-based)",
                       CMAP_BLUES, "Crossing strength")
    _plot_path_matrix(fig2, ax_ca, find_crossings_by_area(), "find_crossings_by_area (coverage overlap)",
                       CMAP_BLUES, "Crossing strength")
    fig2.suptitle("Geometric crossing strength: two ways to compute it")
    fig2.tight_layout()

    # --- 3. adjencency_matrix: all 4 types, same panel/state/frequency ---
    sample_state = int(random_states[0])
    adj_types = ["basic", "peak", "by_area", "geometry"]
    fig3, axes3 = plt.subplots(1, len(adj_types), figsize=(6 * len(adj_types), 6))
    for ax, adj_type in zip(axes3, adj_types):
        adj = adjencency_matrix(st_103, sample_state, freq_i, type=adj_type)
        _plot_path_matrix(fig3, ax, adj, f"type='{adj_type}'", CMAP_SEQUENTIAL, "Adjacency weight")
    fig3.suptitle(f"adjencency_matrix -- Panel 103, state {sample_state}, all 4 types")
    fig3.tight_layout()

    # --- 4. failed_sensor_at: panel 104 has a recorded failure at state 15 (sensor 8) ---
    print("\nfailed_sensor_at('104', state) around its recorded failure threshold (state 15, sensor 8):")
    for s in (10, 14, 15, 16, 20):
        sensor = failed_sensor_at("104", s)
        print(f"  state {s:>2}: {'sensor ' + str(sensor) + ' failed' if sensor else 'no failure yet'}")

    plt.show()


