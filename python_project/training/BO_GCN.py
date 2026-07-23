"""
Bayesian hyperparameter optimization for DeepGraphCNN (models/GCN.py), trained via a
leave-one-out cross-validation loop over BASE_PANELS -- mirrors BO_features.py's
structure (same kt.BayesianOptimization subclass-and-override-run_trial pattern), but
drives a PyTorch/torch_geometric training loop instead of a TF one; KerasTuner's HP
sampling / Bayesian search core doesn't care what run_trial actually does internally.

Hyperparameters tuned:
    - batch_size, learning_rate
    - nr_hidden_channels (0 or 1): whether DeepGraphCNN has one hidden GCNConv layer
      between the fixed 24-dim input compression and the final scalar output layer
    - hidden_dim: width of that hidden layer (only used when nr_hidden_channels == 1,
      but always registered as an hp -- see get_hidden_dim_hp)
    - dropout

Fixed (not tuned):
    - num_node_features: always DEFAULT_GCN_FEATURES (the AE's latent_dim -- every
      Panel_GraphDataset here is built with big_latent=True), compressed to a fixed
      INPUT_COMPRESS_DIM=24 via DeepGraphCNN's input_compress_dim (mirrors GraphCNN's
      own Linear(num_node_features, 24) step)
    - use_residual: always True

Search objective: "Objective" (direction="max") =
    mean_fitness - DAMAGE_LOSS_WEIGHT * mean_damage_loss
where mean_fitness is a*monotonicity + b*trendability + c*prognosability
(prognostic_criteria.py) computed from each fold's trained model's own HI predictions
across CV_PANELS, and mean_damage_loss is GCN_train.damage_map_loss evaluated on that
fold's held-out validation panel. damage_map_loss is unbounded (not a small fixed-scale
criterion like fitness), so DAMAGE_LOSS_WEIGHT is a small fixed weight -- check the
printed mean_damage_loss magnitude for your data and adjust it if one term ends up
silently dominating the other. mean_train_loss/mean_val_loss (global_loss + path_loss,
the actual backprop'd training loss) are also tracked every trial purely as training
diagnostics (plotted, don't drive the search) -- same convention as BO_features.py.

After the search: retrains one model per CV_PANELS fold with the best hyperparameters
(saving each as a GCN_train.py-compatible checkpoint dict), builds + saves an
EnsembleGCN from those folds, and reproduces the same diagnostic plots as
BO_features.py (progress, running best, overfitting, HP sensitivity, HP coverage) plus
GCN_train.py's own training-history / HI plots per fold and for the ensemble.
"""
import sys
from functools import partial
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os

import numpy as np
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt
import torch
import torch_geometric

from GCN import DeepGraphCNN
from graph_dataset import Panel_GraphDataset, features_GraphDataset
from GCN_train import (
    monotonicity_loss, combined_path_loss, damage_map_loss,
    plot_training_history, plot_HI, build_and_save_ensemble, ensemble_predict,
)
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import (
    GRAPH_DATA_DIR, BO_TUNER_DIR, BO_SEARCH_RESULTS_DIR,
    BASE_PANELS, DEFAULT_FREQ_INDEX, DEFAULT_GCN_FEATURES, DEFAULT_N_FEATURES
)

CV_PANELS = [int(p) for p in BASE_PANELS]
TEST_PANEL = 123  # external test panel, mirrors GCN_train.py's own __main__ convention
EPOCHS_PER_FOLD = 200

DAMAGE_LOSS_WEIGHT = 0.01  # small fixed weight; damage_map_loss is unbounded, fitness is ~0-3 -- check magnitudes and adjust
TYPE = "basic"  # default adjacency matrix type for Panel_GraphDataset (see graph_dataset.py) -- "basic", "by_area", "geometry", or "peak"
RAW_FEATURES = True  # default: if True, uses features_GraphDataset instead of Panel_GraphDataset (raw frequency features instead of AE latent features) -- see graph_dataset.py
HP_COLS = ["batch_size", "learning_rate", "nr_hidden_channels", "hidden_dim", "dropout"]


def _input_dims(raw_features):
    """(num_node_features, input_compress_dim) for the given raw_features toggle --
    both are the same value (raw features are wider than the AE latent, so
    "compressing" to that same width is really just a learned Linear+LeakyReLU
    ahead of the GCN stack, not an actual dimensionality reduction). Computed from
    the *argument*, not the RAW_FEATURES global, so build_model/_build_from_params
    build the right-shaped model for whatever a given sweep call actually asked for.
    """
    n = DEFAULT_N_FEATURES if raw_features else DEFAULT_GCN_FEATURES
    return n, n


# ─── HP helpers ────────────────────────────────────────────────────────────────
def get_batch_size_hp(hp):
    return hp.Int("batch_size", min_value=4, max_value=32, step=4)


def get_lr_hp(hp):
    return hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")


def get_nr_hidden_channels_hp(hp, raw_features):
    if raw_features:
        return hp.Fixed("nr_hidden_channels", 1)
    return hp.Int("nr_hidden_channels", min_value=0, max_value=1)


def get_hidden_dim_hp(hp, raw_features):
    # Always registered regardless of nr_hidden_channels's sampled value -- KerasTuner
    # locks an hp's bounds to whatever was seen on its FIRST registration for the whole
    # search (see BO_features.py's get_k_sparse_frac_hp for the same issue hit before).
    # Simply unused in build_model when nr_hidden_channels == 0.
    if raw_features:
        return hp.Int("hidden_dim", min_value=32, max_value=128, step=8)
    return hp.Int("hidden_dim", min_value=8, max_value=64, step=8)

def get_dropout_hp(hp):
    return hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)


# ─── Model builder ──────────────────────────────────────────────────────────────
def build_model(hp, raw_features=RAW_FEATURES):
    nr_hidden = get_nr_hidden_channels_hp(hp, raw_features)
    hidden_dim = get_hidden_dim_hp(hp, raw_features)
    dropout = get_dropout_hp(hp)

    num_node_features, input_compress_dim = _input_dims(raw_features)
    hidden_channels = (hidden_dim,) if nr_hidden == 1 else ()
    return DeepGraphCNN(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        dropout=dropout,
        use_residual=True if nr_hidden == 1 else False,
        input_compress_dim=input_compress_dim,
    )


def _build_from_params(params, raw_features=RAW_FEATURES):
    """Same as build_model, but from a plain {hp_name: value} dict (e.g.
    best.hyperparameters.values) instead of a live kt.HyperParameters object --
    used by the post-search retrain loop."""
    num_node_features, input_compress_dim = _input_dims(raw_features)
    hidden_channels = (params["hidden_dim"],) if params["nr_hidden_channels"] == 1 else ()
    return DeepGraphCNN(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        dropout=params["dropout"],
        use_residual=True if params["nr_hidden_channels"] == 1 else False,
        input_compress_dim=input_compress_dim,
    )


# ─── Fitness (prognostic_criteria.py) ────────────────────────────────────────────
def _collect_HI(model, dataset, device):
    """HI predictions in increasing-state order for one panel's dataset (never
    shuffled here), matching the DI format monotonicity/trendability/prognosability
    criteria expect."""
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=32, shuffle=False)
    all_HI, all_states = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            all_HI.append(HI.cpu().numpy().reshape(-1))
            all_states.append(data.y.cpu().numpy().reshape(-1))
    HI = np.concatenate(all_HI)
    states = np.concatenate(all_states)
    return HI[np.argsort(states)]


def fitness_objective(model, datasets, device, a=1.0, b=1.0, c=1.0):
    """Higher is better. trendability_criterion needs >= 2 panels to correlate
    against each other, so `datasets` should be the full CV panel set."""
    DI = [_collect_HI(model, ds, device) for ds in datasets]
    monotonicity = monotonicity_criterion(DI)
    trendability = trendability_criterion(DI)
    prognosability = prognosability_criterion(DI)
    return a * monotonicity + b * trendability + c * prognosability


# ─── Custom tuner ─────────────────────────────────────────────────────────────
class MyGCNTuner(kt.BayesianOptimization):
    def __init__(self, *args, freq=DEFAULT_FREQ_INDEX, cv_panels=None,
                 epochs_per_fold=EPOCHS_PER_FOLD, type=TYPE, raw_features=RAW_FEATURES, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.cv_panels = cv_panels or CV_PANELS
        self.epochs_per_fold = epochs_per_fold
        self.type = type
        self.raw_features = raw_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = get_batch_size_hp(hp)
        lr = get_lr_hp(hp)

        if self.raw_features:
            datasets = {
                panel: features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=self.freq)
                for panel in self.cv_panels
            }
        else:
            datasets = {
                panel: Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=self.freq, big_latent=True, type=self.type)
                for panel in self.cv_panels
            }

        fold_fitness, fold_damage, fold_val_loss, fold_train_loss = [], [], [], []

        # leave-one-out over cv_panels
        for val_panel in self.cv_panels:
            train_panels = [p for p in self.cv_panels if p != val_panel]
            train_ds = torch.utils.data.ConcatDataset([datasets[p] for p in train_panels])
            train_loader = torch_geometric.loader.DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader = torch_geometric.loader.DataLoader(datasets[val_panel], batch_size=bs, shuffle=False)

            model = self.hypermodel.build(hp).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_loss = float("inf")
            best_train_loss_at_best = float("inf")
            best_state = None
            patience, patience_counter = 10, 0

            for epoch in range(self.epochs_per_fold):
                model.train()
                total_train_loss = 0.0
                n_train_batches = 0
                for data in train_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                    loss = (monotonicity_loss(HI, data.y, data.panel)
                            + combined_path_loss(HI, out, data.y, data.panel)+ damage_map_loss(out, data.y, data.panel))
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    n_train_batches += 1
                avg_train_loss = total_train_loss / max(n_train_batches, 1)

                model.eval()
                total_val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(self.device)
                        out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                        loss = (monotonicity_loss(HI, data.y, data.panel)
                                + combined_path_loss(HI, out, data.y, data.panel)+ damage_map_loss(out, data.y, data.panel))
                        total_val_loss += loss.item()
                        n_val_batches += 1
                avg_val_loss = total_val_loss / max(n_val_batches, 1)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_train_loss_at_best = avg_train_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            fold_val_loss.append(best_val_loss)
            fold_train_loss.append(best_train_loss_at_best)

            # damage_map_loss on the held-out validation panel, using this fold's model
            model.eval()
            total_damage_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.device)
                    out, _ = model(data.x, data.edge_index, data.batch, data.edge_weight)
                    total_damage_loss += damage_map_loss(out, data.y, data.panel).item()
                    n_batches += 1
            fold_damage.append(total_damage_loss / max(n_batches, 1))

            fold_fitness.append(
                fitness_objective(model, [datasets[p] for p in self.cv_panels], self.device)
            )

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_fitness = float(np.mean(fold_fitness))
        mean_damage = float(np.mean(fold_damage))
        objective = mean_fitness - DAMAGE_LOSS_WEIGHT * mean_damage

        return {
            "Objective": objective,
            "mean_fitness": mean_fitness,
            "mean_damage_loss": mean_damage,
            "mean_train_loss": float(np.mean(fold_train_loss)),
            "mean_val_loss": float(np.mean(fold_val_loss)),
        }


# ─── Retrain (best hyperparameters, one model per CV fold) ────────────────────
def _retrain_fold(best_params, val_panel, freq, out_dir, epochs, seed=42, type=TYPE, raw_features=RAW_FEATURES):
    """Retrain one leave-one-out fold with the given (best) hyperparameter values.
    Saves a GCN_train.py-compatible checkpoint dict ({'model','feature_mean',
    'feature_std'}) so build_and_save_ensemble / ensemble_predict / plot_HI all work
    on it unmodified, plus a learning-curve plot via GCN_train.plot_training_history.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    if raw_features:
        datasets = {p: features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq) for p in CV_PANELS}
        test_dataset = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL, freq=freq)
    else:
        datasets = {p: Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq, big_latent=True, type=type) for p in CV_PANELS}
        test_dataset = Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL, freq=freq, big_latent=True, type=type)
    
    train_panels = [p for p in CV_PANELS if p != val_panel]

    # normalization stats from training panels only (mirrors GCN_train.train_with_features)
    all_train_datasets = [datasets[p] for p in train_panels]
    all_features = torch.cat([data.x for ds in all_train_datasets for data in ds], dim=0)
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0)
    feature_std[feature_std < 1e-10] = 1.0

    def norm_transform(data):
        data.x = (data.x - feature_mean) / feature_std
        return data

    for ds in list(datasets.values()) + [test_dataset]:
        ds.transform = norm_transform

    model = _build_from_params(best_params, raw_features=raw_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    train_ds = torch.utils.data.ConcatDataset(all_train_datasets)
    loader = torch_geometric.loader.DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = torch_geometric.loader.DataLoader(datasets[val_panel], batch_size=best_params["batch_size"], shuffle=False)

    loss_history = torch.zeros(epochs)
    global_loss_history = torch.zeros(epochs)
    path_loss_history = torch.zeros(epochs)
    damage_loss_history = torch.zeros(epochs)
    val_loss_history = torch.zeros(epochs)
    global_val_loss_history = torch.zeros(epochs)
    path_val_loss_history = torch.zeros(epochs)
    damage_val_loss_history = torch.zeros(epochs)

    net_name = f"gcn_bo_val_{val_panel}.pt"
    ckpt_path = os.path.join(out_dir, net_name)
    best_val_loss = None
    epochs_done = 0

    for epoch in range(epochs):
        model.train()
        total_loss = total_global = total_path = total_damage = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            global_loss = monotonicity_loss(HI, data.y, data.panel)
            path_loss = combined_path_loss(HI, out, data.y, data.panel)
            damage_loss = damage_map_loss(out, data.y, data.panel)
            loss = global_loss + path_loss + damage_loss 
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); total_global += global_loss.item()
            total_path += path_loss.item(); total_damage += damage_loss.item()
        n = len(loader)
        avg_loss, avg_global, avg_path, avg_damage = total_loss / n, total_global / n, total_path / n, total_damage / n

        model.eval()
        total_vloss = total_vglobal = total_vpath = total_vdamage = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                global_loss = monotonicity_loss(HI, data.y, data.panel)
                path_loss = combined_path_loss(HI, out, data.y, data.panel)
                damage_loss = damage_map_loss(out, data.y, data.panel)
                total_vloss += (global_loss + path_loss + damage_loss).item()
                total_vglobal += global_loss.item(); total_vpath += path_loss.item()
                total_vdamage += damage_loss.item()
        nv = len(val_loader)
        avg_vloss, avg_vglobal = total_vloss / nv, total_vglobal / nv
        avg_vpath, avg_vdamage = total_vpath / nv, total_vdamage / nv

        loss_history[epoch] = avg_loss; global_loss_history[epoch] = avg_global
        path_loss_history[epoch] = avg_path; damage_loss_history[epoch] = avg_damage
        val_loss_history[epoch] = avg_vloss; global_val_loss_history[epoch] = avg_vglobal
        path_val_loss_history[epoch] = avg_vpath; damage_val_loss_history[epoch] = avg_vdamage
        epochs_done += 1

        if best_val_loss is None or avg_vloss < best_val_loss:
            best_val_loss = avg_vloss
            torch.save({'model': model, 'feature_mean': feature_mean, 'feature_std': feature_std}, ckpt_path)

    history_dict = {
        'loss_history': loss_history, 'val_loss_history': val_loss_history,
        'global_loss_history': global_loss_history, 'global_val_loss_history': global_val_loss_history,
        'path_loss_history': path_loss_history, 'path_val_loss_history': path_val_loss_history,
        'damage_loss_history': damage_loss_history, 'damage_val_loss_history': damage_val_loss_history,
    }
    plot_training_history(history_dict, epochs_done,
                           save_dir=os.path.join(out_dir, f"learning_curves_val_{val_panel}.svg"))

    checkpoint = torch.load(ckpt_path, weights_only=False)
    fold_model = checkpoint['model']
    plot_HI(
        fold_model,
        train_datasets=[datasets[p] for p in train_panels],
        validation_datasets=[datasets[val_panel]],
        test_datasets=[test_dataset],
        save_dir=os.path.join(out_dir, f"HI_plot_val_{val_panel}.svg"),
    )
    return net_name


# ─── Helpers for parsing tuner results ────────────────────────────────────────
def _scalar(v):
    if hasattr(v, "value"):
        v = v.value
    if isinstance(v, (list, tuple)):
        return float(v[0])
    return float(v)


def _collect_metric_histories(tuner):
    fitness_h, damage_h, val_h, train_h = [], [], [], []
    for t in tuner.oracle.trials.values():
        if t.score is None:
            continue
        fitness_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_fitness")]
        damage_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_damage_loss")]
        val_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_val_loss")]
        train_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_train_loss")]
        if fitness_obs and damage_obs and val_obs and train_obs:
            fitness_h.append(max(fitness_obs))
            damage_h.append(min(damage_obs))
            val_h.append(min(val_obs))
            train_h.append(min(train_obs))
    return fitness_h, damage_h, train_h, val_h


def _build_trials_dataframe(tuner):
    records = []
    for tid, t in tuner.oracle.trials.items():
        if t.score is None:
            continue
        row = {"trial_id": tid, "objective": t.score}
        row.update(t.hyperparameters.values)

        fitness_hist = [_scalar(o) for o in t.metrics.get_history("mean_fitness")]
        damage_hist = [_scalar(o) for o in t.metrics.get_history("mean_damage_loss")]
        val_hist = [_scalar(o) for o in t.metrics.get_history("mean_val_loss")]
        train_hist = [_scalar(o) for o in t.metrics.get_history("mean_train_loss")]

        row["final_fitness"] = fitness_hist[-1] if fitness_hist else None
        row["final_damage_loss"] = damage_hist[-1] if damage_hist else None
        row["final_val_loss"] = val_hist[-1] if val_hist else None
        row["final_train_loss"] = train_hist[-1] if train_hist else None
        row["overfit_gap"] = (
            row["final_val_loss"] - row["final_train_loss"]
            if (row["final_val_loss"] is not None and row["final_train_loss"] is not None)
            else None
        )
        records.append(row)
    return pd.DataFrame(records).sort_values("objective", ascending=False)


# ─── Plotting (same style/layout as BO_features.py) ────────────────────────────
def _min_max_normalize(values):
    x = np.asarray(values, dtype=float)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def plot_optimization_progress(fitness_h, damage_h, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(_min_max_normalize(fitness_h), label="Fitness (min-max)")
    ax.plot(_min_max_normalize(damage_h), label="Damage map loss (min-max)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Min-Max Normalized [0, 1]")
    ax.set_title("Fitness & Damage Map Loss per Trial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_running_best(objective_h, save_path):
    running_best = np.maximum.accumulate(objective_h) if len(objective_h) else np.array([])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(running_best, "k--", label="Running Best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective")
    ax.set_title("Running Best — GCN BO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return running_best


def plot_overfitting(df, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["final_train_loss"], df["final_val_loss"],
                     c=df["overfit_gap"], cmap="RdYlGn_r", edgecolors="k", s=60)
    fig.colorbar(sc, ax=ax, label="overfit gap (val - train)")
    lo, hi = df["final_train_loss"].min(), df["final_train_loss"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="no overfit line")
    ax.set_xlabel("Final train loss")
    ax.set_ylabel("Final val loss")
    ax.set_title("Overfitting — GCN BO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_hp_sensitivity(df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    for ax, col in zip(axes.flat, HP_COLS):
        ax.scatter(df[col], df["objective"], alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(col); ax.set_ylabel("objective")
        ax.set_title(f"{col} vs objective")
    for ax in axes.flat[len(HP_COLS):]:
        ax.axis("off")
    fig.suptitle("HP Sensitivity — GCN BO", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_hp_coverage(df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    for ax, col in zip(axes.flat, HP_COLS):
        ax.hist(df[col].dropna(), bins=10, edgecolor="k")
        ax.set_title(col)
    for ax in axes.flat[len(HP_COLS):]:
        ax.axis("off")
    fig.suptitle("Sampled HP Distribution — GCN BO", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ─── Main optimization routine ────────────────────────────────────────────────
def run_bayesian_optimization(freq=DEFAULT_FREQ_INDEX, max_trials=30, out_dir=None,
                               epochs_per_fold=EPOCHS_PER_FOLD, retrain_epochs=None, type=TYPE, raw_features=RAW_FEATURES):
    t_start = pd.Timestamp.now()
    tuner = MyGCNTuner(
        partial(build_model, raw_features=raw_features),
        objective=kt.Objective("Objective", direction="max"),
        max_trials=max_trials,
        directory=str(BO_TUNER_DIR),
        project_name="gcn",
        overwrite=True,
        freq=freq,
        epochs_per_fold=epochs_per_fold,
        type=type,
        raw_features=raw_features
    )
    tuner.search()
    t_elapsed = pd.Timestamp.now() - t_start
    print(f"GCN Bayesian optimization completed in {t_elapsed}")

    if type=="basic":

        if raw_features:
            out_dir = str(BO_SEARCH_RESULTS_DIR / f"Bayesian_GCN_raw_freq{freq}") if out_dir is None else out_dir
        else:
            out_dir = str(BO_SEARCH_RESULTS_DIR / f"Bayesian_GCN_freq{freq}") if out_dir is None else out_dir

    else:
        out_dir = str(BO_SEARCH_RESULTS_DIR / f"Bayesian_GCN_{type}_freq{freq}") if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    best = tuner.oracle.get_best_trials(1)[0]
    print("Trial ID:", best.trial_id)
    print("Objective (mean_fitness - damage_weight*mean_damage_loss):", best.score)
    print("Hyperparameters:", best.hyperparameters.values)
    print("Final mean_fitness:", best.metrics.get_last_value("mean_fitness"))
    print("Final mean_damage_loss:", best.metrics.get_last_value("mean_damage_loss"))

    with open(f"{out_dir}/best_trial_details.txt", "w") as f:
        f.write(f"Optimization time {t_elapsed}\n")
        f.write(f"Trial ID: {best.trial_id}\n")
        f.write(f"Objective: {best.score}\n")
        f.write(f"Hyperparameters: {best.hyperparameters.values}\n")
        f.write(f"Final mean_fitness: {best.metrics.get_last_value('mean_fitness')}\n")
        f.write(f"Final mean_damage_loss: {best.metrics.get_last_value('mean_damage_loss')}\n")
        f.write(f"Final mean_train_loss: {best.metrics.get_last_value('mean_train_loss')}\n")
        f.write(f"Final mean_val_loss: {best.metrics.get_last_value('mean_val_loss')}\n")

    # ── Retrain: one model per CV fold with the best hyperparameters ──────────
    best_params = best.hyperparameters.values
    fold_net_names = []
    for val_panel in CV_PANELS:
        net_name = _retrain_fold(best_params, val_panel, freq, out_dir,
                                  epochs=retrain_epochs or epochs_per_fold,
                                  type=type, raw_features=raw_features)
        fold_net_names.append(net_name)

    # ── Ensemble: must run BEFORE build_and_save_ensemble writes ensemble_model.pt
    # into out_dir, since ensemble_predict scans every .pt file in model_path and
    # expects each to be a {'model','feature_mean','feature_std'} checkpoint dict --
    # the ensemble file itself is a raw EnsembleGCN module, not that dict shape.

    if raw_features:
        cv_datasets = [features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq) for p in CV_PANELS]
        test_dataset = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL, freq=freq)
    else:
        cv_datasets = [Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq, big_latent=True, type=type) for p in CV_PANELS]
        test_dataset = Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL, freq=freq, big_latent=True, type=type)
    
    ensemble_predict(cv_datasets, test_dataset, model_path=out_dir,
                      save_dir=os.path.join(out_dir, "ensemble_HI_plot.svg"))

    ensemble_path = os.path.join(out_dir, "ensemble_model.pt")
    build_and_save_ensemble(out_dir, ensemble_path)

    # ── BO diagnostic plots (same style as BO_features.py) ─────────────────────
    fitness_h, damage_h, train_h, val_h = _collect_metric_histories(tuner)
    objective_h = [t.score for t in tuner.oracle.trials.values() if t.score is not None]
    df = _build_trials_dataframe(tuner)
    print(df.head(10))

    plot_optimization_progress(fitness_h, damage_h, f"{out_dir}/bayesian_optimization_progress.svg")
    plot_running_best(objective_h, f"{out_dir}/bayesian_running_best.svg")
    plot_overfitting(df, f"{out_dir}/overfitting_analysis.svg")
    plot_hp_sensitivity(df, f"{out_dir}/hp_sensitivity.svg")
    plot_hp_coverage(df, f"{out_dir}/hp_coverage.svg")

    return {
        "tuner": tuner,
        "best": best,
        "best_params": best_params,
        "fold_checkpoints": fold_net_names,
        "ensemble_path": ensemble_path,
        "fitness_h": fitness_h,
        "damage_h": damage_h,
        "train_h": train_h,
        "val_h": val_h,
        "objective_h": objective_h,
        "df": df,
        "out_dir": out_dir,
    }


if __name__ == "__main__":
    run_bayesian_optimization(max_trials=30)
