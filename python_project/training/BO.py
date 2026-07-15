"""
Bayesian hyperparameter optimization for fc_AE and CNN_AE autoencoders.

Configure MODE / MODEL_TYPE / MAX_TRIALS at the bottom of the file:
    MODE = "single"  -> optimize one model type (uses MODEL_TYPE)
    MODE = "duo"     -> optimize both fc_AE and CNN_AE, plot side-by-side

Hyperparameters for FC_AE:
    - hidden_layer_size1: 512 to 2048 (step 256)
    - hidden_layer_size2: 256 to 1024 (step 128)
    - hidden_layer_size3: 128 to 512 (step 64)
    - hidden_layer_size4: 64 to 256 (step 32)
    - drop_rate: 0.0 to 0.5 (step 0.1)
    - k_sparse: 5 to 15 (step 1)
    - batch_size: 4 to 32 (step 4)

Hyperparameters for CNN_AE:
    - kernel_size1: [50, 20, 10, 5]
    - kernel_size2: [20, 10, 5, 3]
    - filter1:      [32, 64, 128]
    - filter2:      [16, 32, 64]
    - pool_size1:   [20, 10, 5]
    - pool_size2:   [10, 5, 2]
    - batch_size:   [4, 8, 16, 32]

Hardcoded constants:
    - K_SPARSE_PENALTY_WEIGHT: weight for the k-sparse penalty added to the loss
    - TRAIN_DS_NAMES, VAL_DS_NAMES, TEST_DS_NAMES: panel splits for training/validation/testing
    - CV_PANELS: panels used for cross-validation (leave-one-out)
    - EPOCHS_PER_FOLD: number of epochs to train in each fold of cross-validation
    - MODEL_DB_DIR: directory to save the model database Excel file
    - learning_rate: learning rate for training the models (currently fixed at 0.001)
    - loss_weights: weights for the reconstruction loss and latent loss (currently fixed at 0.5 and 1.0)
    - Test set is always 123

At the end train and validation datasets are assumed and latent representations and signal reconstructions are ploted. 
The final reported loss is the mean of the penalized validation loss across the cross-validation folds.

"""

import gc
import os
import sys
from functools import partial
from pathlib import Path
import psutil, os

# for checking memory usage during optimization
def log_mem(tag=""):
    rss = psutil.Process(os.getpid()).memory_info().rss
    print(f"[MEM {tag}] {rss / 1e9:.2f} GB")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from big_train import model_train, monotonicity_loss
from fc_AE import (
    build_CNN_variable_block,
    build_deep_fully_connected_network,
)
from results_viz import PlotContext, plot_sHI_vs_RUL
from states_check import prepare_datastores
from config import (
    K_SPARSE_PENALTY_WEIGHT, TRAIN_PANELS, VAL_PANELS, VAL_123_SUBPANELS,
    TEST_123_SUBPANELS, BASE_PANELS, BO_RESULTS_DIR, BO_TUNER_DIR, BO_SEARCH_RESULTS_DIR,
)


# ─── Constants ────────────────────────────────────────────────────────────────

# HP columns reported in sensitivity / coverage plots, per model type.
HP_COLS = {
    # Fully connected AE has the following HP
    "fc_AE": [
        "hidden_layer_size1", "hidden_layer_size2", "hidden_layer_size3",
        "hidden_layer_size4", "drop_rate", "k_sparse", "batch_size",
    ],
    # CNN AE has the following HP
    "CNN_AE": [
        "kernel_size1", "kernel_size2", "pool_size1", "pool_size2",
        "filter1", "filter2", "batch_size",
    ],
}

# Train / val / test split — these can be used if you don't want to do cross-validation
TRAIN_DS_NAMES = TRAIN_PANELS
VAL_DS_NAMES = VAL_PANELS
TEST_DS_NAMES = VAL_123_SUBPANELS + TEST_123_SUBPANELS

CV_PANELS = BASE_PANELS   # leave-one-out folds
EPOCHS_PER_FOLD = 50

MODEL_DB_DIR = str(BO_RESULTS_DIR)


# ─── HP helpers & k-sparse penalty ────────────────────────────────────────────
def k_sparse_penalty(k, weight=K_SPARSE_PENALTY_WEIGHT):
    return float(weight) * float(k)


def get_k_sparse_hp(hp):
    return hp.Int("k_sparse", min_value=5, max_value=15, step=1)


def get_batch_size_hp(hp):
    return hp.Int("batch_size", min_value=4, max_value=32, step=4)


class KSparsePenaltyCallback(tf.keras.callbacks.Callback):
    """Adds penalized loss metrics to logs at end of each epoch."""

    def __init__(self, k_sparse, weight=K_SPARSE_PENALTY_WEIGHT):
        super().__init__()
        self.k_sparse = int(k_sparse)
        self.weight = float(weight)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        penalty = k_sparse_penalty(self.k_sparse, self.weight)
        if "loss" in logs:
            logs["loss_penalized"] = float(logs["loss"]) + penalty
        if "val_loss" in logs:
            logs["val_loss_penalized"] = float(logs["val_loss"]) + penalty


# ─── Model builder ────────────────────────────────────────────────────────────
def build_model(hp, model_type="fc_AE"):
    if model_type == "fc_AE":
        params = {
            "hidden_layer_size1": hp.Int("hidden_layer_size1", 512, 2048, step=256),
            "hidden_layer_size2": hp.Int("hidden_layer_size2", 256, 1024, step=128),
            "hidden_layer_size3": hp.Int("hidden_layer_size3", 128, 512, step=64),
            "hidden_layer_size4": hp.Int("hidden_layer_size4", 64, 256, step=32),
            "desired_latent_size": 15,
            "drop_rate": hp.Float("drop_rate", 0.0, 0.5, step=0.1),
            "k_sparse": get_k_sparse_hp(hp),
            "input_size": 4000,
            "num_in": 1,
        }
        model = build_deep_fully_connected_network(params)
    elif model_type == "CNN_AE":
        params = {
            "kernel_size1": hp.Choice("kernel_size1", values=[50, 20, 10, 5]),
            "kernel_size2": hp.Choice("kernel_size2", values=[20, 10, 5, 3]),
            "filter1":      hp.Choice("filter1",      values=[32, 64, 128]),
            "filter2":      hp.Choice("filter2",      values=[16, 32, 64]),
            "pool_size1":   hp.Choice("pool_size1",   values=[20, 10, 5]),
            "pool_size2":   hp.Choice("pool_size2",   values=[10, 5, 2]),
            "n_blocks": 2,
        }
        model = build_CNN_variable_block(params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Output naming differs between models:
    #   fc_AE  : reconstruction layer is "fc_output_1"
    #   CNN_AE : reconstruction layer is "final_1"  (after crop/pad)
    recon_name = "final_1" if model_type == "CNN_AE" else "fc_output_1"
    model.compile(
        optimizer=optimizer,
        loss={recon_name: "mse", "fc_latent_1": monotonicity_loss},
        loss_weights={recon_name: 0.5, "fc_latent_1": 1.0},
    )
    return model


# ─── Custom tuner ─────────────────────────────────────────────────────────────
class MyTuner(kt.BayesianOptimization):
    def __init__(self, *args, path_i, freq, model_type=None, 
                 cv_panels=None, epochs_per_fold=EPOCHS_PER_FOLD, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.path_i = path_i
        self.freq = freq
        self.cv_panels = cv_panels or CV_PANELS
        self.epochs_per_fold = epochs_per_fold

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = get_batch_size_hp(hp)
        k_sparse = get_k_sparse_hp(hp)

        fold_val, fold_val_pen = [], []
        fold_tr,  fold_tr_pen  = [], []

        # run cross-validation folds for every panel in the cv_panels
        for fold_idx, val_panel in enumerate(self.cv_panels):

            log_mem(f"trial {trial.trial_id} fold {fold_idx} before prep")

            train_panels = [p for p in self.cv_panels if p != val_panel]
            print(f"\n[trial {trial.trial_id}] fold {fold_idx+1}/"
                  f"{len(self.cv_panels)} — val panel {val_panel}")

            train_ds, val_ds, *_ = prepare_datastores(
                path_i=self.path_i,
                freq_i=self.freq,
                base_batch_size=bs,
                test_batch_size=1,
                train_ds_names=train_panels,
                val_ds_names=[val_panel],
                test_ds_names=TEST_DS_NAMES,
                include_benchmark=(self.model_type == "CNN_AE"),
            )

            log_mem(f"trial {trial.trial_id} fold {fold_idx} after prep")

            # Fresh, compiled model for this fold
            model = self.hypermodel.build(hp)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs_per_fold,
                callbacks=[
                    KSparsePenaltyCallback(k_sparse=k_sparse),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=4,
                        min_lr=1e-5,
                    ),
                ],
                verbose=0,
            )

            h = history.history
            fold_val.append(min(h.get("val_loss", [float("inf")])))
            fold_val_pen.append(min(h.get("val_loss_penalized", [float("inf")])))
            fold_tr.append(min(h.get("loss", [float("inf")])))
            fold_tr_pen.append(min(h.get("loss_penalized", [float("inf")])))

            # release graph/memory between folds
            del model, history, train_ds, val_ds, h
            tf.keras.backend.clear_session()
            gc.collect()
            log_mem(f"trial {trial.trial_id} fold {fold_idx} END") 

        log_mem(f"trial {trial.trial_id} END") 
        # Returning a dict tells KerasTuner: "these are the metrics for this trial"
        return {
            "mean_val_loss":            float(np.mean(fold_val)),
            "mean_val_loss_penalized":  float(np.mean(fold_val_pen)),
            "std_val_loss_penalized":   float(np.std(fold_val_pen)),
            "mean_train_loss":          float(np.mean(fold_tr)),
            "mean_train_loss_penalized":float(np.mean(fold_tr_pen)),
        }


# ─── Helpers for parsing tuner results ────────────────────────────────────────
def _scalar(v):
    """Coerce a KerasTuner MetricObservation / array / tensor to a python float."""
    if hasattr(v, "value"):
        v = v.value
    if isinstance(v, tf.Tensor):
        v = v.numpy()
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0])
    if isinstance(v, (list, tuple)):
        return float(v[0])
    return float(v)


def _collect_loss_histories(tuner):
    val_h, train_h = [], []
    for t in tuner.oracle.trials.values():
        if t.score is None:  # skip incomplete trials
            continue
        val_obs   = [_scalar(o.value) for o in t.metrics.get_history("mean_val_loss_penalized")]
        train_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_train_loss_penalized")]
        if val_obs and train_obs:
            val_h.append(min(val_obs))
            train_h.append(min(train_obs))
    return train_h, val_h


def _build_trials_dataframe(tuner):
    records = []
    for tid, t in tuner.oracle.trials.items():
        if t.score is None:
            continue
        row = {"trial_id": tid, "objective": t.score}
        row.update(t.hyperparameters.values)

        val_hist       = [_scalar(o) for o in t.metrics.get_history("mean_val_loss")]
        train_hist     = [_scalar(o) for o in t.metrics.get_history("mean_train_loss")]
        val_pen_hist   = [_scalar(o) for o in t.metrics.get_history("mean_val_loss_penalized")]
        train_pen_hist = [_scalar(o) for o in t.metrics.get_history("mean_train_loss_penalized")]

        row["final_val_loss"]             = val_hist[-1]       if val_hist       else None
        row["final_train_loss"]           = train_hist[-1]     if train_hist     else None
        row["final_val_loss_penalized"]   = val_pen_hist[-1]   if val_pen_hist   else None
        row["final_train_loss_penalized"] = train_pen_hist[-1] if train_pen_hist else None
        row["overfit_gap"] = (
            row["final_val_loss"] - row["final_train_loss"]
            if (row["final_val_loss"] is not None
                and row["final_train_loss"] is not None)
            else None
        )
        records.append(row)
    return pd.DataFrame(records).sort_values("objective")


def _save_best_trial_details(best, t_elapsed, out_dir):
    with open(f"{out_dir}/best_trial_details.txt", "w") as f:
        f.write(f"Optimization time {t_elapsed}\n")
        f.write(f"Trial ID: {best.trial_id}\n")
        f.write(f"Objective (val_loss_penalized): {best.score}\n")
        f.write(f"Hyperparameters: {best.hyperparameters.values}\n")
        f.write(f"Final val_loss: {best.metrics.get_last_value('mean_val_loss')}\n")
        f.write(f"Final val_loss_penalized: {best.metrics.get_last_value('mean_val_loss_penalized')}\n")
        f.write(f"Final train_loss: {best.metrics.get_last_value('mean_train_loss')}\n")
        f.write(f"Train loss history: {best.metrics.get_history('mean_train_loss')}\n")
        f.write(f"Train loss penalized history: {best.metrics.get_history('mean_train_loss_penalized')}\n")


def _append_to_model_database(model_info, results_dir=MODEL_DB_DIR):
    os.makedirs(results_dir, exist_ok=True)
    excel_path = os.path.join(results_dir, "model_database.xlsx")
    df_new = pd.DataFrame([model_info])

    if not os.path.exists(excel_path):
        df_new.to_excel(excel_path, index=False)
        print(f"Created new database: {excel_path}")
        return

    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        try:
            existing_df = pd.read_excel(excel_path)
            df_new.to_excel(writer, index=False, header=False,
                            startrow=len(existing_df) + 1)
        except Exception:
            df_new.to_excel(excel_path, index=False)
    print(f"Updated database at {excel_path}")

# ─── Main optimization routine ────────────────────────────────────────────────
def run_bayesian_optimization(path_i, freq, model_type="fc_AE", max_trials=30, out_dir=None, db_dir=None):
    """Run Bayesian search and train the best model.

    Returns a dict with:
        model_type, best_model, best_params,
        train_h, val_h, running_best, df, out_dir


    """
    if model_type == "fc_AE":
        model_building_function = build_model
    elif model_type == "CNN_AE":
        model_building_function = partial(build_model, model_type="CNN_AE")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    t_start = pd.Timestamp.now()
    tuner = MyTuner(
        model_building_function,
        objective=kt.Objective("mean_val_loss_penalized", direction="min"),
        max_trials=max_trials,
        directory=str(BO_TUNER_DIR),
        project_name="ae",
        overwrite=True,
        model_type=model_type,
        path_i=path_i,
        freq=freq
    )
    tuner.search()
    t_elapsed = pd.Timestamp.now() - t_start
    print(f"Bayesian optimization ({model_type}) completed in {t_elapsed}")

    date = pd.Timestamp.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_dir = str(BO_SEARCH_RESULTS_DIR / f"Bayesian_{model_type}_{date}") if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    best = tuner.oracle.get_best_trials(1)[0]
    print("Trial ID:", best.trial_id)
    print("Objective (val_loss_penalized):", best.score)
    print("Hyperparameters:", best.hyperparameters.values)
    print("Final val_loss:", best.metrics.get_last_value("mean_val_loss"))
    print("Final val_loss_penalized:", best.metrics.get_last_value("mean_val_loss_penalized"))
    print("Final train loss:", best.metrics.get_last_value("mean_train_loss"))

    _save_best_trial_details(best, t_elapsed, out_dir)

    # Train best model with a fresh seed and append to the model database.
    best_params = best.hyperparameters.values
    seed = int(np.random.randint(0, 10000))
    tf.random.set_seed(seed)
    best_model, _, _, model_info, *_ = model_train(
        model_type, params=best_params, seed=seed, n_blocks=2, path_i=path_i, frequency_i=freq, results_dir=out_dir, epochs=EPOCHS_PER_FOLD
    )
    
    _append_to_model_database(model_info, results_dir=db_dir or MODEL_DB_DIR)

    train_h, val_h = _collect_loss_histories(tuner)
    running_best = np.minimum.accumulate(val_h) if val_h else np.array([])
    df = _build_trials_dataframe(tuner)
    print(df.head(10))

    return {
        "model_type": model_type,
        "best_model": best_model,
        "best_params": best_params,
        "train_h": train_h,
        "val_h": val_h,
        "running_best": running_best,
        "df": df,
        "out_dir": out_dir,
    }


# ─── Plotting (works for 1 or N models — single-mode is just N=1) ─────────────
def plot_optimization_progress(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(r["val_h"],   label="Validation Loss")
        ax.plot(r["train_h"], label="Training Loss")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best Loss")
        ax.set_title(f"Best Loss per Trial — {r['model_type']}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)


def plot_running_best(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(r["running_best"], "k--", label="Running Best")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best Validation Loss")
        ax.set_title(f"Running Best — {r['model_type']}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)


def plot_overfitting(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, r in zip(axes[0], results):
        df = r["df"]
        sc = ax.scatter(df["final_train_loss"], df["final_val_loss"],
                        c=df["overfit_gap"], cmap="RdYlGn_r",
                        edgecolors="k", s=60)
        fig.colorbar(sc, ax=ax, label="overfit gap (val - train)")
        lo, hi = df["final_train_loss"].min(), df["final_train_loss"].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="no overfit line")
        ax.set_xlabel("Final train loss")
        ax.set_ylabel("Final val loss")
        ax.set_title(f"Overfitting — {r['model_type']}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)


def plot_hp_sensitivity(result, save_path):
    """Per-model HP scatter (HP cols differ per model, so this is single-model)."""
    df = result["df"]
    cols = HP_COLS[result["model_type"]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        ax.scatter(df[col], df["objective"], alpha=0.6,
                   edgecolors="k", linewidths=0.3)
        ax.set_xlabel(col); ax.set_ylabel("objective")
        ax.set_title(f"{col} vs objective")
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"HP Sensitivity — {result['model_type']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)


def plot_hp_coverage(result, save_path):
    df = result["df"]
    cols = HP_COLS[result["model_type"]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna(), bins=10, edgecolor="k")
        ax.set_title(col)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"Sampled HP Distribution — {result['model_type']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)


def plot_all(results):
    """Generate every diagnostic plot.

    Combined plots (progress / running_best / overfitting) go into the first
    result's out_dir. HP sensitivity / coverage are per-model, into each
    result's own out_dir.
    """
    primary_dir = results[0]["out_dir"]
    plot_optimization_progress(results, f"{primary_dir}/bayesian_optimization_progress.png")
    plot_running_best(results,         f"{primary_dir}/bayesian_running_best.png")
    plot_overfitting(results,          f"{primary_dir}/overfitting_analysis.png")
    for r in results:
        plot_hp_sensitivity(r, f"{r['out_dir']}/hp_sensitivity.png")
        plot_hp_coverage(r,    f"{r['out_dir']}/hp_coverage.png")


# ─── Predictions on best model ────────────────────────────────────────────────
def run_predictions(result, path_i, freq_i, state_idx=0,):
    """sHI-vs-RUL predictions for the best model in a result bundle."""
    model_type = result["model_type"]
    is_cnn = model_type == "CNN_AE"


    train_names = TRAIN_DS_NAMES 
    val_names = VAL_DS_NAMES

    _, _, _, ds_dict, target_dict, RUL_dict, States_dict, _ = prepare_datastores(
        path_i=path_i,
        freq_i=freq_i,
        base_batch_size=result["best_params"].get("batch_size"),
        test_batch_size=1,
        train_ds_names=train_names,
        val_ds_names=val_names,
        test_ds_names=TEST_DS_NAMES,
        include_benchmark=is_cnn,
    )
    ctx = PlotContext(
        ds_dict=ds_dict,
        target_dict=target_dict,
        RUL_dict=RUL_dict,
        train_ds_names=train_names,
        validation_ds_names=val_names,
        test_ds_names=TEST_DS_NAMES,
        States_dict=States_dict,
    )

    suffix = "_bench" if is_cnn else ""
    plot_sHI_vs_RUL(result["best_model"], ctx, state_idx, "train",
                    save=True, dir=result["out_dir"],
                    plot_name=f"train_val_sHI_vs_RUL{suffix}", show=False)
    plot_sHI_vs_RUL(result["best_model"], ctx, state_idx, "validation",
                    save=True, dir=result["out_dir"],
                    plot_name=f"val_sHI_vs_RUL{suffix}", show=False)


# ─── Entry point ──────────────────────────────────────────────────────────────
# Configure the run here:
MODE = "single"            # "single" or "duo"
MODEL_TYPE = "CNN_AE"    # used only when MODE == "single"  ("fc_AE" or "CNN_AE")
MAX_TRIALS = 30
PATH_I = 0
FREQ_I = 3


def main():
    # cleanup from any previous runs (in case we're running multiple times in a row)
    plt.close("all")
    tf.keras.backend.clear_session()

    folder_name = f"Multi_path_{pd.Timestamp.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    
    for PATH_I in range(28,29):
        out_dir = f"{folder_name}/Bayesian_{MODEL_TYPE}_path{PATH_I}"
        if MODE == "single":
            results = [run_bayesian_optimization(PATH_I, FREQ_I, MODEL_TYPE, max_trials=MAX_TRIALS, out_dir=out_dir, db_dir=folder_name)]
        elif MODE == "duo":
            results = [
                run_bayesian_optimization(PATH_I, FREQ_I, "fc_AE",  max_trials=MAX_TRIALS, out_dir=out_dir, db_dir=folder_name),
                run_bayesian_optimization(PATH_I, FREQ_I, "CNN_AE", max_trials=MAX_TRIALS, out_dir=out_dir, db_dir=folder_name),
            ]
        else:
            raise ValueError(f"Unknown MODE: {MODE!r}. Use 'single' or 'duo'.")

        plot_all(results)
        for r in results:
            run_predictions(r, PATH_I, FREQ_I)

    #plt.show()

        # ── cleanup before next path ─────────────────────
        plt.close("all")           # release matplotlib figures
        del results                # drop refs to model, ds_dict, df, etc.
        tf.keras.backend.clear_session()
        gc.collect()
        log_mem(f"after path {PATH_I} cleanup")  # verify it actually drops


if __name__ == "__main__":
    main()