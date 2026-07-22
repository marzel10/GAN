"""
Central configuration for python_project.

Consolidates paths, panel/sensor constants, and shared hyperparameter
defaults that were previously duplicated -- and had drifted -- across many
files. Every entry below is commented with which file(s) currently own or
duplicate it, so call sites can be swapped over one at a time.
"""

from itertools import combinations
from pathlib import Path

import numpy as np

# ===========================================================================
# Paths
# ===========================================================================
# Anchored to the physical location of this file, not the caller's cwd.
# The old code used relative path strings ("graph_data", "GCN_models", ...)
# which silently created duplicate folders depending on which directory a
# script happened to be launched from (repo root vs python_project/). That
# is why, e.g., model_train_results/ and model_database.xlsx used to exist
# in two different places on disk.
PROJECT_ROOT = Path(__file__).resolve().parent  # .../GAN/python_project/

DATA_DIR = PROJECT_ROOT / "data"                                          # raw States_<panel>.mat files. Read by: states.py, states_check.py
FEATURES_CACHE_DIR = PROJECT_ROOT / "features_cache"                      # Written by: features_extractor.py (extract_all_features). Read by: graph_dataset.py
GRAPH_DATA_DIR = PROJECT_ROOT / "graph_data"                              # PyG dataset root (raw/ + processed/). Written/read by: graph_dataset.py, GCN_train.py, imagining_alghoritm.py, extract_shi.py, make_animations.py, prognostic_criteria.py
GCN_MODELS_DIR = PROJECT_ROOT / "GCN_models"                              # Written by: GCN_train.py (train, train_with_features). Read by: imagining_alghoritm.py
MODELS_FEATURES_DIR = PROJECT_ROOT / "models_features"                   # Written by: big_train.py (model_train_features, per-path .h5 checkpoints). Read by: heatmap_AE.py
ANIMATIONS_DIR = PROJECT_ROOT / "animations"                              # Default output_dir in imagining_alghoritm.py (animate_panel / animate_panel_sidebyside). NOTE: dead default -- every current caller overrides output_dir explicitly
CROSS_VALIDATION_RESULTS_DIR = PROJECT_ROOT / "Cross_Validation_Results"  # Written by: GCN_train.py (__main__ cross-validation block). Read by: make_animations.py
CROSS_VALIDATION_RESULTS_AE_DIR = PROJECT_ROOT / "Cross_Validation_Results_AE"  # Written by: big_train.py (__main__ cross-validation block)
MODEL_TRAIN_RESULTS_DIR = PROJECT_ROOT / "model_train_results"            # Written by: big_train.py (model_train, model_train_features)
BO_RESULTS_DIR = PROJECT_ROOT / "BO_results"                              # Written by: BO.py (_append_to_model_database default results_dir)
BO_TUNER_DIR = PROJECT_ROOT / "tuner_dir"                                 # keras-tuner scratch dir. Written by: BO.py (MyTuner directory=)
BO_SEARCH_RESULTS_DIR = PROJECT_ROOT / "results"                         # Written by: BO.py (run_bayesian_optimization default out_dir)
MODEL_DATABASE_XLSX = PROJECT_ROOT / "model_database.xlsx"                # Appended to by: big_train.py, BO.py (_append_to_model_database)
MODEL_DATABASE_XLSX_FEATURES = PROJECT_ROOT / "model_database_features.xlsx"  # Appended to by: big_train.py (model_train_features)
ARCHIVE_DIR = PROJECT_ROOT / "archive"                                    # Historical experiment runs (Model_for_every_path/, results_hype_tweaked/, old model_database.xlsx backups). Not written to by any current code.


def mat_file_path(panel_name: str) -> Path:
    """e.g. mat_file_path('103') -> DATA_DIR / 'States_103.mat'.

    Replaces path-building duplicated in: states_check.py, graph_dataset.py,
    features_extractor.py, weight_matrix.py, visualize_crossing_graph.py,
    weird_123_behaviour.py.
    """
    return DATA_DIR / f"States_{panel_name}.mat"


# ===========================================================================
# Panels
# ===========================================================================
BASE_PANELS = ["103", "104", "105", "109"]  # Previously redefined independently in: BO.py, big_train.py, GCN_train.py, extract_shi.py, states_check.py, states_plot.py
PANEL_123_SUBPANELS = ["123_1", "123_2", "123_31", "123_32", "123_41", "123_42", "123_43", "123_44"]  # Previously redefined in: BO.py, big_train.py, results_viz.py, graph_dataset.py, features_extractor.py, states_check.py
ALL_PANELS = BASE_PANELS + ["123"] + PANEL_123_SUBPANELS

TRAIN_PANELS = ["103", "104", "105"]                          # Used by: BO.py (TRAIN_DS_NAMES), big_train.py (train_ds_names default), results_viz.py (train_ds_names)
VAL_PANELS = ["109"]                                          # Used by: BO.py (VAL_DS_NAMES), big_train.py (val_ds_names default), results_viz.py (val_ds_names)
TEST_PANELS = PANEL_123_SUBPANELS                             # Used by: BO.py (TEST_DS_NAMES), big_train.py (test_ds_names default), results_viz.py (test_ds_names) -- all 8 "123" subpanels held out as the test set
VAL_123_SUBPANELS = ["123_1", "123_31", "123_41", "123_43"]   # states_check.py val split (finer split: half of the 123 subpanels)
TEST_123_SUBPANELS = ["123_2", "123_32", "123_42", "123_44"]  # states_check.py test split (finer split: the other half of the 123 subpanels)
CV_PANELS = ["103", "104", "109", "105"]


# Per-123-subpanel saved-state counts / global start offsets.
# Previously byte-for-byte duplicated in weight_matrix.py and graph_dataset.py.
NR_SAVED_STATES = {"123_1": 16, "123_2": 7, "123_31": 16, "123_32": 13, "123_41": 10, "123_42": 10, "123_43": 10, "123_44": 10}
STATE_START_INDICES = {"123_1": 0, "123_2": 16, "123_31": 23, "123_32": 39, "123_41": 52, "123_42": 62, "123_43": 72, "123_44": 82}

TOTAL_STATES = {"103": 32, "104": 58, "105": 30, "109": 28, "123": 92}  # Owned by: GCN_train.py

# Sensor-failure bookkeeping: {panel_name: [relative_state_index, failed_sensor_index]}.
# Owned by: weight_matrix.py; consumed elsewhere only via its failed_sensor_at() helper.
FAILURES_RECORD = {
    "123_42": [9, 3],
    "104": [15, 8],
    "105": [16, 8],
}

VALIDATION_PANEL_MAP = {"123": "109"}  # Owned by: extract_shi.py

# ===========================================================================
# Frequency / sampling
# ===========================================================================
DEFAULT_FREQ_INDEX = 2  # Hardcoded as freq=2 independently in: damage_map.py, extract_shi.py, heatmap_AE.py, states_check.py, states_plot.py, features_extractor.py, results_viz.py, big_train.py, GCN_train.py, graph_dataset.py, imagining_alghoritm.py, make_animations.py, prognostic_criteria.py, visualize_crossing_graph.py
FREQUENCY_MAPPING = [50,100,125,150,200,250] # [kHz]
# ===========================================================================
# Sensor / panel geometry (moved from plot_panel.py)
# ===========================================================================
PANEL_W = 0.165  # metres. Owned by: plot_panel.py
PANEL_H = 0.240  # metres. Owned by: plot_panel.py

SENSOR_POSITIONS = np.array([
    [0.025,  0.025],   # S1
    [0.100,  0.025],   # S2
    [0.140,  0.215],   # S3
    [0.065,  0.215],   # S4
    [0.025,  0.120],   # S5
    [0.140,  0.120],   # S6
    [0.0825, 0.080],   # S7
    [0.0825, 0.160],   # S8
])  # Owned by: plot_panel.py. Imported by: GCN_train.py, imagining_alghoritm.py, visualize_crossing_graph.py, weight_matrix.py, heatmap_AE.py, make_animations.py

SENSOR_PAIRS = [(i, j) for i, j in combinations(range(1, 9), 2)]  # 28 unique 1-indexed sensor pairs. Owned by: plot_panel.py

DAMAGE_POINTS = {
    103: np.array([0.050,  0.080]),
    104: np.array([0.025,  0.080]),
    105: np.array([0.115,  0.160]),  # (165-50, 240-80) mm
    109: np.array([0.0825, 0.140]),  # (165-82.5, 240-100) mm
    123: np.array([[0.0825 - 0.03, 0.045], [0.0825, 0.045], [0.0825 - 0.03, 0.045 + 0.03], [0.0825, 0.045 + 0.03]]),  # square debond surface
}  # Owned by: plot_panel.py

# ===========================================================================
# Physical / WCPDI algorithm constants (moved from imagining_alghoritm.py)
# ===========================================================================
WAVE_VELOCITY = 1560         # v. Owned by: imagining_alghoritm.py
TIME_CONST = 0.001   # T. Owned by: imagining_alghoritm.py
MAX_DIST = WAVE_VELOCITY * TIME_CONST  # Imported by: GCN_train.py

DEFAULT_WCPDI_C = 0.9      # Repeated as a bare literal in: heatmap_AE.py, imagining_alghoritm.py, GCN_train.py, make_animations.py
DEFAULT_WCPDI_BETA = 0.5   # Repeated as a bare literal in: GCN_train.py, make_animations.py
DEFAULT_N_PIXELS = 1000000   # Repeated as a bare literal in: GCN_train.py, make_animations.py

# ===========================================================================
# Model hyperparameter defaults
# ===========================================================================
DEFAULT_K_SPARSE = 10          # Repeated independently 11x across fc_AE.py and big_train.py
DEFAULT_N_FEATURES = 66        # Duplicated as N_FEAT in big_train.py and the n_feat default in fc_AE.py
K_SPARSE_PENALTY_WEIGHT = 0.01  # Owned by: BO.py

# ===========================================================================
# Colormaps
# ===========================================================================
CMAP_HEATMAP = "hot"          # heatmap_AE.py, imagining_alghoritm.py
CMAP_SEQUENTIAL = "viridis"   # damage_map.py, weight_matrix.py, visualize_crossing_graph.py
CMAP_DIVERGING = "plasma"     # plot_panel.py, damage_map.py
CMAP_BLUES = "Blues"          # visualize_crossing_graph.py, weight_matrix.py
