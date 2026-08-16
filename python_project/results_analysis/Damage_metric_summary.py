'''
This script calculates the mean difference between the damage at the impact location
and the mean damage of the whole panel, for a trained model's own health-index
predictions (an evaluation-time, non-differentiable counterpart to
GCN_train.damage_map_loss -- see also imagining_alghoritm.animate_panel_sidebyside,
which plots the same two quantities over states for a single GCN model/panel).

It takes the following arguments (see main() / CLI below):
    - model type: one of "basic", "by_area", "geometry", "peak", "wml" (GCN adjacency
      types, see training/sweep_over_options.py) or "path" (per-path AE ensemble, see
      path_performance.py)
    - panel number: 103, 104, 105, 109, or 123
    - beta constant: spatial-weighting beta passed into P_AE/U (imagining_alghoritm.py).
      Also selects the trained checkpoint for model_type="by_area", the only GCN type
      beta_constant was swept for (see training/sweep_over_options.run_by_area_beta_sweep)
    - frequency of the signal: index 0-5 into config.FREQUENCY_MAPPING
    - life fraction to take into account (default is 1.0): only the first
      round(life_fraction * n_states) states (oldest/healthiest first) are averaged over

For the requested model type, this obtains the ensemble model's per-path prediction for
every one of the panel's paths (28) at every recorded state -- for GCN types this means
loading the saved ensemble_model.pt checkpoint and re-running it (only the graph-level HI
is cached in graph_performance_results_<type>/HI.pkl, not the per-path breakdown); for
"path" it is read directly from path_performance_results/sHI.pkl, which already caches
that per-path array for the ensemble fold.

For every state (or every state within the life fraction), the 28 per-path values are
combined into a WCPDI spatial map (same P/U/WCPDI weighting scheme as
imagining_alghoritm.py), from which the damage value at the panel's known impact/damage
point (config.DAMAGE_POINTS) and the mean damage over the whole panel are read off. The
difference between those two quantities (damage_at_point - panel_mean) is then averaged
over the selected states.
'''
import argparse
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric

from graph_dataset import Panel_GraphDataset, features_GraphDataset
from imagining_alghoritm import P_AE, U, WCPDI
from config import (
    BASE_PANELS, BETA_CONSTANT, BETAS, BO_SEARCH_RESULTS_DIR, DAMAGE_POINTS, FREQ_FOR_BETA_SWEEP,
    FREQUENCY_MAPPING, GRAPH_DATA_DIR, PANEL_W, PANEL_H, PROJECT_ROOT, GCN_TYPES, MODEL_TYPES, TYPES_LABELS, PANELS, DAMAGE_MAP_N_PIXELS, COMPARE_OUT_DIR,
    _LINESTYLES, CUSTOM_PALETTE as PALETTE, COMPARE_OUT_DIR, LIFETIME_FRACTIONS
)


def _beta_suffix(beta):
    # Mirrors BO_GCN.run_bayesian_optimization's own out_dir naming.
    return f"_beta{beta}" if beta != BETA_CONSTANT else ""


def _gcn_folder(model_type, freq, beta):
    '''Bayesian_GCN_* checkpoint folder on disk for a (type, frequency, beta) combo --
    mirrors BO_GCN.run_bayesian_optimization's out_dir naming.'''
    if model_type == "basic":
        return f"Bayesian_GCN_freq{freq}"
    if model_type == "wml":
        
        return f"Bayesian_GCN_without_map_loss_freq{freq}"
    if model_type == "by_area":
        return f"Bayesian_GCN_by_area_freq{freq}{_beta_suffix(beta)}"

    
    return f"Bayesian_GCN_{model_type}_freq{freq}"


def _adjacency_type(model_type):
    return "basic" if model_type == "wml" else model_type


def _collect_path_out(model, dataset, device):
    '''Per-path (node-level) outputs for every state in `dataset`, sorted into
    increasing-state order. Returns (n_states, N_PATHS).'''
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=32, shuffle=False)
    all_out, all_states = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, data.edge_weight)
            batch_size = data.num_graphs
            num_paths = out.shape[0] // batch_size
            all_out.append(out.reshape(batch_size, num_paths).cpu().numpy())
            all_states.append(data.y.cpu().numpy().reshape(-1))
    out_arr = np.concatenate(all_out, axis=0)
    states = np.concatenate(all_states)
    return out_arr[np.argsort(states)]


def _load_gcn_path_out(model_type, panel_number, freq, beta):
    '''(n_states, N_PATHS) per-path ensemble-model outputs for one GCN model type /
    panel / frequency / beta, recomputed from the saved ensemble_model.pt checkpoint
    (only the graph-level HI, not the per-path breakdown, is cached on disk for GCN
    types -- see graph_performance.py's HI.pkl).'''
    folder = BO_SEARCH_RESULTS_DIR / _gcn_folder(model_type, freq, beta)
    ensemble_path = folder / "ensemble_model.pt"
    if not ensemble_path.exists():
        raise FileNotFoundError(f"{ensemble_path} not found -- has this model been trained?")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model = torch.load(ensemble_path, map_location=device, weights_only=False).to(device)
    dataset = Panel_GraphDataset(
        root=str(GRAPH_DATA_DIR), panel_number=panel_number, freq=freq,
        big_latent=True, type=_adjacency_type(model_type), beta_constant=beta,
    )
    return _collect_path_out(ensemble_model, dataset, device)

def _load_raw_gcn_path_out(model_type, panel_number, freq, beta):
    '''(n_states, N_PATHS) per-path ensemble-model outputs for one GCN model type /
    panel / frequency / beta, recomputed from the saved ensemble_model.pt checkpoint
    (only the graph-level HI, is cached on disk for GCN
    types -- see graph_performance.py's HI.pkl).'''
    folder = BO_SEARCH_RESULTS_DIR / _gcn_folder(model_type, freq, beta)
    ensemble_path = folder / "ensemble_model.pt"
    if not ensemble_path.exists():
        raise FileNotFoundError(f"{ensemble_path} not found -- has this model been trained?")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model = torch.load(ensemble_path, map_location=device, weights_only=False).to(device)
    dataset = features_GraphDataset(
        root=str(GRAPH_DATA_DIR), panel_number=panel_number, freq=freq
    )
    return _collect_path_out(ensemble_model, dataset, device)

def _load_path_ae_out(panel_number, freq):
    '''(n_states, N_PATHS) ensemble-fold sHI curves for the per-path AE, read straight
    from path_performance_results/sHI.pkl (sHI[fold][freq][panel][path] -- already
    cached at every state for the ensemble fold by path_performance.py).'''
    sHI_path = PROJECT_ROOT / "path_performance_results" / "sHI.pkl"
    if not sHI_path.exists():
        raise FileNotFoundError(f"{sHI_path} not found -- run path_performance.py first")
    with open(sHI_path, "rb") as f:
        sHI = pickle.load(f)

    fold_keys = BASE_PANELS + ["ensemble"]
    panels = BASE_PANELS + ["123"]
    fold_idx = fold_keys.index("ensemble")
    panel_idx = panels.index(str(panel_number))

    per_path = sHI[fold_idx][freq][panel_idx]  # length-28 list of 1-D arrays (state order)
    missing = [i for i, p in enumerate(per_path) if p is None]
    if missing:
        raise ValueError(f"panel={panel_number} freq={freq}: missing sHI for paths {missing}")
    return np.stack(per_path, axis=1)  # (n_states, N_PATHS)


def load_per_path_out(model_type, panel_number, freq, beta=BETA_CONSTANT):
    '''(n_states, N_PATHS) per-path health-index curve (ensemble model/fold) for the
    requested model type/panel/frequency/beta. GCN types are recomputed from the saved
    ensemble checkpoint; "path" reads the already-cached per-path AE ensemble sHI.'''
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}, expected one of {MODEL_TYPES}")
    if beta != BETA_CONSTANT and model_type not in ("by_area", "path"):
        raise ValueError(
            f"beta_constant={beta} requested but no beta-specific checkpoint exists for "
            f"model_type={model_type!r} -- beta_constant was only swept for 'by_area' "
            f"(see training/sweep_over_options.run_by_area_beta_sweep); pass "
            f"beta={BETA_CONSTANT} (the default) for other model types."
        )
    if model_type == "path":
        return _load_path_ae_out(panel_number, freq)

    if model_type == "raw":
        return _load_raw_gcn_path_out(model_type, panel_number, freq, beta)
    
    return _load_gcn_path_out(model_type, panel_number, freq, beta)


def damage_vs_mean_per_state(per_path_out, panel_number, beta=BETA_CONSTANT, n_pixels=DAMAGE_MAP_N_PIXELS):
    '''For every state, builds the WCPDI spatial map from that state's 28 per-path values'''
    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    damage_point = DAMAGE_POINTS[panel_number]
    if damage_point.ndim == 2:
        damage_point = damage_point.mean(axis=0)
    ix = int(np.clip(round(damage_point[0] / dx), 0, X.shape[0] - 1))
    iy = int(np.clip(round(damage_point[1] / dx), 0, X.shape[1] - 1))

    n_states = per_path_out.shape[0]
    damage_vals = np.empty(n_states)
    mean_vals = np.empty(n_states)
    for state in range(n_states):
        U_arr = np.zeros_like(X)
        U(U_arr, (X, Y), panel_number=panel_number, state=state, beta_constant=beta)
        P_arr = np.zeros_like(X)
        P_AE(P_arr, (X, Y), per_path_out[state], panel_number=panel_number, state=state, beta_constant=beta)
        WCPDI_map = WCPDI(P_arr, U_arr)
        damage_vals[state] = WCPDI_map[ix, iy]
        mean_vals[state] = np.nanmean(WCPDI_map)

    return damage_vals, mean_vals


def evaluate_damage_loss(model_type, panel_number, freq, beta=BETA_CONSTANT, life_fraction=1.0,
                          n_pixels=DAMAGE_MAP_N_PIXELS, recompute=True):
    '''Compute damage map metrics for a given model type/panel/frequency/beta, to avoid recomputing the per-path outputs every time metric is stored in a cache'''
    cache_path = COMPARE_OUT_DIR / f"damage_metrics_{model_type}_panel{panel_number}_freq{freq}_beta{beta}_pixels{n_pixels}.pkl"
    if cache_path.exists() and not recompute:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        damage_vals, mean_vals, diff = cached["damage_vals"], cached["mean_vals"], cached["diff"]
    else:
        per_path_out = load_per_path_out(model_type, panel_number, freq, beta=beta)
        damage_vals, mean_vals = damage_vs_mean_per_state(per_path_out, panel_number, beta=beta, n_pixels=n_pixels)
        diff = (damage_vals - mean_vals) / np.abs(mean_vals)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"damage_vals": damage_vals, "mean_vals": mean_vals, "diff": diff}, f)

    n_states = damage_vals.shape[0]
    
    n_keep = max(1, int(round(life_fraction * n_states)))
    return {
        "damage_vals": damage_vals,
        "mean_vals": mean_vals,
        "diff": diff,
        "n_states": n_states,
        "n_states_used": n_keep,
        "mean_diff": float(np.nanmean(diff[:n_keep])),
    }

FREQ_LABELS = [f"{f} kHz" for f in FREQUENCY_MAPPING] + ["average"]
PANEL_LABELS = {103: "L1-03", 104: "L1-04", 105: "L1-05", 109: "L1-09", 123: "L1-23 (test)", "average": "Val. Avg."}

def _palette_color(i):
    return PALETTE[i % len(PALETTE)]

def _palette_style(i):
    color = PALETTE[i % len(PALETTE)]
    linestyle = _LINESTYLES[(i // len(PALETTE)) % len(_LINESTYLES)]
    return color, linestyle


def _plot_damage_loss_vs_freq(data, panel_cols, title, file_name, out_dir=COMPARE_OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(FREQ_LABELS))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, col in enumerate(panel_cols):
        is_avg = col == "average"
        marker_sign = "o"if col != 123 else "s"
        color, linestyle = _palette_style(i)
        ax.plot(x, data[:, i], linestyle="--" if is_avg else linestyle, marker=marker_sign, markersize=4,
                color="k" if is_avg else color, label=PANEL_LABELS.get(col, str(col)))
    ax.set_xticks(x)
    ax.set_xticklabels(FREQ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Damage map metrics")
    #ax.set_title(title)
    ax.legend(title="Coupon", fontsize=8, handlelength=4, loc="best")
    ax.grid(True)
    fig.tight_layout()
    save_path = out_dir / file_name
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def compare_GCN_and_AE(out_dir=COMPARE_OUT_DIR):
    '''Compute damage metric averaged over validation panels for every GCN type and the
    per-path AE ensemble, vs frequency, and compare to the held-out test panel.'''
    n_freq = len(FREQUENCY_MAPPING) + 1  # +1 for the average-over-frequencies row
    n_panels = len(PANELS) + 1  # +1 for the average-over-(base)panels column

    damage_loss_GCN = np.empty((n_freq, n_panels))
    damage_loss_AE = np.empty((n_freq, n_panels))
    for freq in range(len(FREQUENCY_MAPPING)):
        for i, panel in enumerate(PANELS + [123]):  # last column recomputed as the average below
            result_GCN = evaluate_damage_loss("basic", panel, freq)
            damage_loss_GCN[freq, i] = result_GCN["mean_diff"]

            result_AE = evaluate_damage_loss("path", panel, freq)
            damage_loss_AE[freq, i] = result_AE["mean_diff"]

    # mean over frequency for every coupon 
    mean_freq_GCN = np.nanmean(damage_loss_GCN[:-1, :], axis=0)
    mean_freq_AE = np.nanmean(damage_loss_AE[:-1, :], axis=0)
    damage_loss_GCN[-1, :] = mean_freq_GCN
    damage_loss_AE[-1, :] = mean_freq_AE

    # mean over valition coupons for every frequency 
    mean_val_GCN = np.nanmean(damage_loss_GCN[:, :-2], axis=1)  # 123 is the test panel so it is excluded
    mean_val_AE = np.nanmean(damage_loss_AE[:, :-2], axis=1)
    damage_loss_GCN[:, -1] = mean_val_GCN
    damage_loss_AE[:, -1] = mean_val_AE


    panel_cols = PANELS + ["average"]
    _plot_damage_loss_vs_freq(damage_loss_GCN, panel_cols, "Damage loss vs frequency (GCN, basic)", "damage_loss_GCN.svg", out_dir=out_dir)
    _plot_damage_loss_vs_freq(damage_loss_AE, panel_cols, "Damage loss vs frequency (Path AE ensemble)", "damage_loss_AE.svg", out_dir=out_dir)

    return damage_loss_GCN, damage_loss_AE


def _plot_damage_loss_vs_type(data_by_type, title, file_name, out_dir=COMPARE_OUT_DIR, std_by_type=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(GCN_TYPES))

    val_avg = np.array([data_by_type[t][-1, -1] for t in GCN_TYPES])
    test_panel = np.array([data_by_type[t][-1, PANELS.index(123)] for t in GCN_TYPES])
    val_avg_std = np.array([std_by_type[t][-1] for t in GCN_TYPES]) if std_by_type else None
    test_panel_std = np.array([std_by_type[t][PANELS.index(123)] for t in GCN_TYPES]) if std_by_type else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, val_avg, yerr=val_avg_std, fmt="o-", markersize = 5 ,capsize=3, color=_palette_color(0), label="Val. Avg.")
    ax.errorbar(x, test_panel, yerr=test_panel_std, fmt="s-", markersize = 5 ,capsize=3, color=_palette_color(1), label="Test (L1-23)")


    points = np.concatenate([val_avg, test_panel])
    y_lo, y_hi = np.nanmin(points), np.nanmax(points)
    pad = 0.1 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    #ax.set_ylim(, y_hi + pad)

    ax.set_xticks(x)
    ax.set_xticklabels([TYPES_LABELS.get(t, t) for t in GCN_TYPES], rotation=45, ha="right")
    #ax.set_xlabel("GCN type")
    ax.set_ylabel("Damage map metrics avg over frequency")
    #ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_path = out_dir / file_name
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def compare_GCN_types(out_dir=COMPARE_OUT_DIR):
    '''Compute damage metric averaged over validation panels for every GCN adjacency-matrix type, vs frequency, and compare to the held-out test panel.'''
    n_freq = len(FREQUENCY_MAPPING) + 1  # +1 for the average-over-frequencies row
    n_panels = len(PANELS) + 1  # +1 for the average-over-(base)panels column

    damage_loss_GCN_types = {model_type: np.empty((n_freq, n_panels)) for model_type in GCN_TYPES}
    for freq in range(len(FREQUENCY_MAPPING)):
        for i, panel in enumerate(PANELS + [123]):  # last column recomputed as the average below
            for model_type in GCN_TYPES:
                result_GCN = evaluate_damage_loss(model_type, panel, freq)
                damage_loss_GCN_types[model_type][freq, i] = result_GCN["mean_diff"]

    damage_loss_GCN_types_std_freq = {}
    for model_type in GCN_TYPES:
        mean_val = np.nanmean(damage_loss_GCN_types[model_type][:-1, :-2], axis=1)  # 123 is the test panel so it is excluded
        std_val = np.nanstd(damage_loss_GCN_types[model_type][:-1, :-2], axis=1)
        damage_loss_GCN_types[model_type][:-1, -1] = mean_val

        mean_freq = np.nanmean(damage_loss_GCN_types[model_type][:-1, :], axis=0)
        std_freq = np.nanstd(damage_loss_GCN_types[model_type][:-1, :], axis=0)
        damage_loss_GCN_types[model_type][-1, :] = mean_freq
        damage_loss_GCN_types_std_freq[model_type] = std_freq
        

    _plot_damage_loss_vs_type(damage_loss_GCN_types, "Damage loss vs GCN type", "damage_loss_GCN_types.svg",
                               out_dir=out_dir, std_by_type=damage_loss_GCN_types_std_freq)
    

    return damage_loss_GCN_types

def _plot_damage_loss_vs_beta(data_by_beta, title, file_name, out_dir=COMPARE_OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    betas = sorted(data_by_beta.keys())
    x = np.arange(len(betas))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, col in enumerate(PANELS + ["average"]):
        is_avg = col == "average"
        marker_sign = "o"if col != 123 else "s"
        color, linestyle = _palette_style(i)
        ax.plot(x, [data_by_beta[beta][i] for beta in betas], linestyle="--" if is_avg else linestyle, marker=marker_sign, markersize=4,
                color="k" if is_avg else color, label=PANEL_LABELS.get(col, str(col)))
    ax.set_xticks(x)
    ax.set_xticklabels([str(beta) for beta in betas], rotation=45, ha="right")
    ax.set_xlabel(r"Beta constant $\frac{1}{f}$")
    ax.set_ylabel("Damage map metrics")
    #ax.set_title(title)
    ax.legend(title="Coupon", fontsize=8, loc="best", handlelength=4)
    ax.grid(True)
    fig.tight_layout()
    save_path = out_dir / file_name
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def compare_betas(freq,out_dir=COMPARE_OUT_DIR):
    '''Compute damage metric averaged over validation panels for the "by_area" GCN type, vs beta constant, and compare to the held-out test panel.'''
    n_panels = len(PANELS) + 1  # +1 for the average-over-(base)panels column
    betas = BETAS 

    damage_loss_betas = {beta: np.empty((n_panels,)) for beta in betas}
    for i, panel in enumerate(PANELS + [123]):  # last column recomputed as the average below
        for beta in betas:
            result_GCN = evaluate_damage_loss("by_area", panel, freq, beta=beta)
            damage_loss_betas[beta][i] = result_GCN["mean_diff"]

    for beta in betas:
        # 123 is the test panel so it is excluded from the validation-panel average.
        damage_loss_betas[beta][-1] = np.nanmean(damage_loss_betas[beta][:-2])

    _plot_damage_loss_vs_beta(damage_loss_betas, "Damage loss vs Beta", "damage_loss_betas_wo5000.svg", out_dir=out_dir)

def _plot_damage_loss_raw_vs_freq(val_avg, val_std, test_panel, types, title, file_name, out_dir=COMPARE_OUT_DIR):
    '''One figure: mean(damage_at_impact_point - panel_mean) vs frequency (6 raw +
    average, categorical x-axis), two lines per type -- validation-panel average
    (BASE_PANELS, error bars = std across those panels) and the held-out test panel
    (123, no error bars, one panel only).'''
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(FREQ_LABELS))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, type_ in enumerate(types):
        color = _palette_color(i)
        if type_ != "raw":
            label = "Processed by AE (Val. Avg.)"
            label_test = "Processed by AE (Test)"
        else: 
            label = "Raw"
            label_test = "Raw (Test)"
        ax.errorbar(x, val_avg[type_], yerr=val_std[type_], fmt="o-", capsize=3,
                    color=color, label=label)
        ax.plot(x, test_panel[type_], "s--", color=color, label=label_test)

    ax.set_ylim(-1,1)
    ax.set_xticks(x)
    ax.set_xticklabels(FREQ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Damage map metrics")
    #ax.set_title(title)
    ax.legend(title="Features:", fontsize=8, loc="best")
    ax.grid(True)
    fig.tight_layout()
    save_path = out_dir / file_name
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def compare_raw(out_dir=COMPARE_OUT_DIR):
    '''Compute damage metric averaged over validation panels for the "raw" GCN type, vs frequency, and compare to the "basic" GCN type.'''
    n_freq = len(FREQUENCY_MAPPING) + 1  # +1 for the average-over-frequencies row
    types_to_show = ["basic",  "raw"]
    base_panels = PANELS[:-1]  # BASE_PANELS as ints -- PANELS' last entry (123) is the test panel

    val_avg = {t: np.empty(n_freq) for t in types_to_show}
    val_std = {t: np.empty(n_freq) for t in types_to_show}
    test_panel = {t: np.empty(n_freq) for t in types_to_show}

    for freq in range(len(FREQUENCY_MAPPING)):
        for type_ in types_to_show:
            base_vals = np.array([evaluate_damage_loss(type_, p, freq)["mean_diff"] for p in base_panels])
            val_avg[type_][freq] = np.nanmean(base_vals)
            val_std[type_][freq] = np.nanstd(base_vals)
            test_panel[type_][freq] = evaluate_damage_loss(type_, 123, freq)["mean_diff"]

    for type_ in types_to_show:
        val_avg[type_][-1] = np.nanmean(val_avg[type_][:-1])
        val_std[type_][-1] = np.nanmean(val_std[type_][:-1])
        test_panel[type_][-1] = np.nanmean(test_panel[type_][:-1])

    _plot_damage_loss_raw_vs_freq(val_avg, val_std, test_panel, types_to_show,
                                   "Damage loss vs frequency (raw GCN)", "damage_loss_raw_cropped.svg", out_dir=out_dir)



def _plot_damage_loss_vs_life_fraction(mean_by_lf, std_by_lf, life_fractions, title, file_name, out_dir=COMPARE_OUT_DIR):
    '''Damage loss metrics averaged over frequency, vs life fraction (x-axis) -- one line per panel, error bars = std over frequency'''
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, panel in enumerate(PANELS):
        means = [mean_by_lf[lf][i] for lf in life_fractions]
        stds = [std_by_lf[lf][i] for lf in life_fractions]
        color, linestyle = _palette_style(i)
        marker_sign = "o" if panel != 123 else "s"
        ax.errorbar(life_fractions, means, yerr=stds, fmt=f"{marker_sign}{linestyle}", capsize=3,
                    color=color, label=PANEL_LABELS.get(panel, str(panel)))

    ax.set_ylim(-5, 5)
    
    ax.set_xlabel("Life fraction")
    ax.set_ylabel("Damage map metrics avg over frequency")
    #ax.set_title(title)
    ax.legend(title="Coupon", fontsize=8, loc="best", handlelength=4)
    ax.grid(True)
    fig.tight_layout()
    save_path = out_dir / file_name
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def compare_life_fractions(model_type, beta=BETA_CONSTANT, n_pixels=DAMAGE_MAP_N_PIXELS, out_dir=COMPARE_OUT_DIR):
    '''Compute mean and std of the damage metric (averaged over frequency),across life fractions (1.0, 0.75, 0.5, 0.25, 0) for one (model_type, beta), one line per panel.'''
    life_fractions = LIFETIME_FRACTIONS
    raw_results = {lf: np.empty((len(FREQUENCY_MAPPING), len(PANELS))) for lf in life_fractions}
    for lf in life_fractions:
        for freq in range(len(FREQUENCY_MAPPING)):
            for i, panel_number in enumerate(PANELS):
                result = evaluate_damage_loss(
                    model_type, panel_number, freq, beta=beta,
                    life_fraction=lf, n_pixels=n_pixels,
                )
                raw_results[lf][freq, i] = result["mean_diff"]

    mean_by_lf = {lf: np.nanmean(raw_results[lf], axis=0) for lf in life_fractions}
    std_by_lf = {lf: np.nanstd(raw_results[lf], axis=0) for lf in life_fractions}

    _plot_damage_loss_vs_life_fraction(
        mean_by_lf, std_by_lf, life_fractions,
        f"Damage loss vs life fraction ({model_type}, beta={beta})",
        f"damage_loss_vs_life_fraction_{model_type}_beta{beta}_cropped.svg", out_dir=out_dir,
    )

    return mean_by_lf, std_by_lf

def compare_peak(out_dir=COMPARE_OUT_DIR):
    '''Compute damage metric averaged over validation panels for the "peak" GCN type, vs frequency'''
    n_freq = len(FREQUENCY_MAPPING) + 1  # +1 for the average-over-frequencies row
    n_panels = len(PANELS) + 1  # +1 for the average-over-(base)panels column

    damage_loss_GCN = np.empty((n_freq, n_panels))
    for freq in range(len(FREQUENCY_MAPPING)):
        for i, panel in enumerate(PANELS + [123]):  # last column recomputed as the average below
            result_GCN = evaluate_damage_loss("peak", panel, freq)
            damage_loss_GCN[freq, i] = result_GCN["mean_diff"]


    # mean over frequency for every coupon 
    mean_freq_GCN = np.nanmean(damage_loss_GCN[:-1, :], axis=0)
    damage_loss_GCN[-1, :] = mean_freq_GCN

    # mean over valition coupons for every frequency 
    mean_val_GCN = np.nanmean(damage_loss_GCN[:, :-2], axis=1)  # 123 is the test panel so it is exclude
    damage_loss_GCN[:, -1] = mean_val_GCN
    


    panel_cols = PANELS + ["average"]
    _plot_damage_loss_vs_freq(damage_loss_GCN, panel_cols, "Damage loss vs frequency (GCN, basic)", "damage_loss_peak.svg", out_dir=out_dir)
    
    return damage_loss_GCN

def main():
    compare_betas(FREQ_FOR_BETA_SWEEP)   
    compare_GCN_types()
    compare_GCN_and_AE()
    compare_peak()
    compare_raw()
    compare_life_fractions("basic")
    compare_life_fractions("peak")
    
if __name__ == "__main__":
    main()
    