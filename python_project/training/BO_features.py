"""
Bayesian hyperparameter optimization for fc_AE and CNN_AE autoencoders trained on time-frequency
features (build_fc_AE_features / build_CNN_AE_features in fc_AE.py -- not the raw-signal
build_deep_fully_connected_network / build_CNN_variable_block architectures).

Configure MODE / MODEL_TYPE / MAX_TRIALS at the bottom of the file:
    MODE = "single"  -> optimize one model type (uses MODEL_TYPE)
    MODE = "duo"     -> optimize both fc_AE and CNN_AE, plot side-by-side

Hyperparameters for FC_AE:
    - latent_dim, k_sparse (bounded by latent_dim), drop_rate, batch_size

Hyperparameters for CNN_AE:
    - latent_dim, k_sparse (bounded by latent_dim), drop_rate, filters_bench, filters_path, batch_size

Hardcoded constants:
    - K_SPARSE_PENALTY_WEIGHT: weight for the k-sparse penalty added to the loss (diagnostic only, see below)
    - TRAIN_DS_NAMES, VAL_DS_NAMES, TEST_DS_NAMES: panel splits for training/validation/testing
    - CV_PANELS: panels used for cross-validation (leave-one-out)
    - EPOCHS_PER_FOLD: number of epochs to train in each fold of cross-validation
    - MODEL_DB_DIR: directory to save the model database Excel file
    - learning_rate: learning rate for training the models (currently fixed at 0.001)
    - loss_weights: weights for the reconstruction loss and latent loss (currently fixed at 1.0 and 2.0)
    - Test set is always 123

Search objective: mean_fitness (direction="max"), computed per CV fold as
a*monotonicity + b*trendability + c*prognosability (see fitness_objective, and
prognostic_criteria.py for the three underlying criteria) from each fold's trained
model's own sHI predictions across CV_PANELS. This replaced the original
mean_val_loss_penalized objective, which is still computed and returned every trial
purely as a training diagnostic (plotted, but no longer drives the search) -- note its
magnitude isn't directly comparable across different batch_size values or training
epochs, since monotonicity_loss sums (not averages) an unbounded per-pair term over the
batch, so it scales with batch_size and with how monotonic the current predictions
happen to be, independent of overall model quality.

At the end train and validation datasets are assumed and latent representations and signal reconstructions are ploted.
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

from big_train import model_train_features, monotonicity_loss
from fc_AE import (
    build_CNN_AE_features,
    build_fc_AE_features,
)
from ae_cross_validation_helper import (
    plot_sHI_cv_fold, plot_reconstruction_cv_fold, build_ensemble_ae, plot_ensemble_sHI,
)
from states_check import prepare_datastores
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import (
    K_SPARSE_PENALTY_WEIGHT, TRAIN_PANELS, VAL_PANELS, VAL_123_SUBPANELS,
    TEST_123_SUBPANELS, BASE_PANELS, BO_RESULTS_DIR, BO_TUNER_DIR, BO_SEARCH_RESULTS_DIR,
    DEFAULT_N_FEATURES,
)


# ─── Constants ────────────────────────────────────────────────────────────────

# HP columns reported in sensitivity / coverage plots, per model type.
# These names match build_fc_AE_features / build_CNN_AE_features's own params exactly.
# CNN_AE_features's two-stage encoder (bench_comp mixes channels, dir_comp pairs each
# stat's half1/half2 values) has its own filters_bench/filters_path, not a single
# "filters"/"kernel_size" pair.
HP_COLS = {
    # Fully connected AE has the following HP
    "fc_AE": [
         "k_sparse_frac", "batch_size", "latent_dim", "drop_rate"
    ],
    # CNN AE has the following HP
    "CNN_AE": [
        "k_sparse_frac", "filters_bench", "filters_path", "latent_dim", "drop_rate", "batch_size"
    ],
}

# Feature vectors are fixed-width (33 features per state, see FeaturesExtractor); not tuned.
N_FEAT = DEFAULT_N_FEATURES

# Train / val / test split — these can be used if you don't want to do cross-validation
TRAIN_DS_NAMES = TRAIN_PANELS
VAL_DS_NAMES = VAL_PANELS
TEST_DS_NAMES = VAL_123_SUBPANELS + TEST_123_SUBPANELS

CV_PANELS = BASE_PANELS   # leave-one-out folds
EPOCHS_PER_FOLD = 100

MODEL_DB_DIR = str(BO_RESULTS_DIR)


# ─── HP helpers & k-sparse penalty ────────────────────────────────────────────
def k_sparse_penalty(k, weight=K_SPARSE_PENALTY_WEIGHT):
    return float(weight) * float(k)


def get_latent_dim_hp(hp):
    return hp.Int("latent_dim", min_value=4, max_value=24, step=4)


def get_k_sparse_frac_hp(hp):
    # A fraction of latent_dim, not an absolute count: KerasTuner locks a hyperparameter's
    # registered bounds to whatever was seen on its *first* registration for the whole
    # search (typically trial 1's "default configuration", which evaluates every hp at
    # its min_value) -- so an hp.Int("k_sparse", max_value=latent_dim-1, ...) with a
    # max_value that depends on latent_dim's *sampled* value doesn't actually vary
    # trial-to-trial; it stays locked at whatever latent_dim's default (min_value) gave
    # on trial 1. A fixed-range fraction sidesteps that entirely -- see resolve_k_sparse.
    return hp.Float("k_sparse_frac", min_value=0.1, max_value=0.9, step=0.1)


def resolve_k_sparse(k_sparse_frac, latent_dim):
    # KSparse.call does k = min(self.k, n_units), so any k_sparse >= latent_dim silently
    # collapses to "no sparsification" -- keep this below latent_dim explicitly.
    return max(2, min(latent_dim - 1, round(k_sparse_frac * latent_dim)))


def get_drop_rate_hp(hp):
    return hp.Float("drop_rate", 0.0, 0.5, step=0.1)


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
    # Shared across both architectures -- sampled once per trial; build_model may be
    # called multiple times (once per CV fold) with the same hp object, and KerasTuner
    # caches by name, so repeat calls with identical bounds just reuse the same value.
    latent_dim = get_latent_dim_hp(hp)
    k_sparse = resolve_k_sparse(get_k_sparse_frac_hp(hp), latent_dim)
    drop_rate = get_drop_rate_hp(hp)

    if model_type == "fc_AE":
        params = {
            "input_size": N_FEAT,
            "n_features": N_FEAT,
            "latent_dim": latent_dim,
            "k_sparse": k_sparse,
            "drop_rate": drop_rate,
        }
        model = build_fc_AE_features(params)
    elif model_type == "CNN_AE":
        params = {
            "n_features": N_FEAT,
            # MyTuner requests include_benchmark=True for CNN_AE (self.benchmark), so
            # prepare_datastores hands back 2-channel (signal, benchmark) data -- must
            # match here or the Input layer's shape won't line up with the datastore.
            "n_channels": 2,
            "latent_dim": latent_dim,
            "k_sparse": k_sparse,
            "drop_rate": drop_rate,
            "filters_bench": hp.Int("filters_bench", min_value=2, max_value=8, step=2),
            "filters_path": hp.Int("filters_path", min_value=2, max_value=12, step=2),
        }
        model = build_CNN_AE_features(params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Both build_fc_AE_features and build_CNN_AE_features always name their outputs
    # "reconstruction" and "sHI" -- no per-architecture branching needed here.
    model.compile(
        optimizer=optimizer,
        loss={"reconstruction": "mse", "sHI": monotonicity_loss},
        loss_weights={"reconstruction": 1.0, "sHI": 2.0},
    )
    return model


# ─── Fitness objective (prognostic_criteria.py) ───────────────────────────────
# fitness_criterion's own graph=False path (prepare_DI_from_AE -> extract_shi) reloads
# saved model files from disk by scanning a folder, and doesn't even use the `model`
# argument it's given -- it can't target one specific in-memory trial's model. So this
# recomputes the same combination (a*monotonicity + b*trendability + c*prognosability)
# directly from the model that's actually in memory for this trial/fold, reusing the
# three underlying criterion functions (which are generic -- they just take a list of
# per-panel sHI arrays, regardless of where those came from).
def _collect_sHI(model, ds_dict, panels):
    """sHI predictions per panel, in state order (ds_dict entries are never shuffled --
    see states_check.prepare_datastores), matching the DI format the criterion
    functions expect: DI[i] = sHI values in increasing-state order for panel i."""
    shi_idx = model.output_names.index("sHI")
    DI = []
    for panel in panels:
        panel_shi = []
        for x, _ in ds_dict[panel]:
            pred = model.predict(x, verbose=0)
            panel_shi.append(np.asarray(pred[shi_idx]).reshape(-1))
        DI.append(np.concatenate(panel_shi))
    return DI


def fitness_objective(model, ds_dict, panels, a=1.0, b=1.0, c=1.0):
    """Higher is better. trendability_criterion needs >=2 panels to correlate against
    each other, so `panels` should be the full CV panel set, not a single held-out one."""
    DI = _collect_sHI(model, ds_dict, panels)
    monotonicity = monotonicity_criterion(DI)
    trendability = trendability_criterion(DI)
    prognosability = prognosability_criterion(DI)
    return a * monotonicity + b * trendability + c * prognosability


# ─── Custom tuner ─────────────────────────────────────────────────────────────
class MyTuner(kt.BayesianOptimization):
    def __init__(self, *args, path_i, freq, model_type=None, 
                 cv_panels=None, epochs_per_fold=EPOCHS_PER_FOLD, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.benchmark = True if model_type == "CNN_AE" else False
        self.diff_bench = True if model_type == "fc_AE" else False
        self.path_i = path_i
        self.freq = freq
        self.cv_panels = cv_panels or CV_PANELS
        self.epochs_per_fold = epochs_per_fold

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = get_batch_size_hp(hp)
        latent_dim = get_latent_dim_hp(hp)
        k_sparse = resolve_k_sparse(get_k_sparse_frac_hp(hp), latent_dim)

        fold_val, fold_val_pen = [], []
        fold_tr,  fold_tr_pen  = [], []
        fold_fitness = []

        # run cross-validation folds for every panel in the cv_panels
        for fold_idx, val_panel in enumerate(self.cv_panels):

            log_mem(f"trial {trial.trial_id} fold {fold_idx} before prep")

            train_panels = [p for p in self.cv_panels if p != val_panel]
            print(f"\n[trial {trial.trial_id}] fold {fold_idx+1}/"
                  f"{len(self.cv_panels)} — val panel {val_panel}")

            train_ds, val_ds, _, ds_dict, *_ = prepare_datastores(
                path_i=self.path_i,
                freq_i=self.freq,
                base_batch_size=bs,
                test_batch_size=1,
                train_ds_names=train_panels,
                val_ds_names=[val_panel],
                test_ds_names=TEST_DS_NAMES,
                include_benchmark=self.benchmark,
                diff_bench=self.diff_bench,
                features=True,
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

            # ds_dict covers every panel in self.cv_panels (not just this fold's val_panel),
            # normalized with this fold's own training stats -- exactly what the model
            # that was just trained on this fold expects as input.
            fold_fitness.append(fitness_objective(model, ds_dict, self.cv_panels))

            # release graph/memory between folds
            del model, history, train_ds, val_ds, ds_dict, h
            tf.keras.backend.clear_session()
            gc.collect()
            log_mem(f"trial {trial.trial_id} fold {fold_idx} END")

        log_mem(f"trial {trial.trial_id} END")
        # Returning a dict tells KerasTuner: "these are the metrics for this trial".
        # mean_fitness is the search objective (see kt.Objective in run_bayesian_optimization);
        # the loss metrics stay purely diagnostic (still plotted, no longer drive the search).
        return {
            "mean_fitness":              float(np.mean(fold_fitness)),
            "std_fitness":               float(np.std(fold_fitness)),
            "mean_val_loss":             float(np.mean(fold_val)),
            "mean_val_loss_penalized":   float(np.mean(fold_val_pen)),
            "std_val_loss_penalized":    float(np.std(fold_val_pen)),
            "mean_train_loss":           float(np.mean(fold_tr)),
            "mean_train_loss_penalized": float(np.mean(fold_tr_pen)),
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
    val_h, train_h, fitness_h = [], [], []
    for t in tuner.oracle.trials.values():
        if t.score is None:  # skip incomplete trials
            continue
        val_obs     = [_scalar(o.value) for o in t.metrics.get_history("mean_val_loss_penalized")]
        train_obs   = [_scalar(o.value) for o in t.metrics.get_history("mean_train_loss_penalized")]
        fitness_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_fitness")]
        if val_obs and train_obs and fitness_obs:
            val_h.append(min(val_obs))
            train_h.append(min(train_obs))
            # fitness is maximized (unlike the loss diagnostics above), so track this
            # trial's best fitness observation with max, not min.
            fitness_h.append(max(fitness_obs))
    return train_h, val_h, fitness_h


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
    # objective is now mean_fitness (higher is better) -- best trials first.
    return pd.DataFrame(records).sort_values("objective", ascending=False)


def _save_best_trial_details(best, t_elapsed, out_dir):
    with open(f"{out_dir}/best_trial_details.txt", "w") as f:
        f.write(f"Optimization time {t_elapsed}\n")
        f.write(f"Trial ID: {best.trial_id}\n")
        f.write(f"Objective (mean_fitness): {best.score}\n")
        f.write(f"Hyperparameters: {best.hyperparameters.values}\n")
        f.write(f"Final fitness: {best.metrics.get_last_value('mean_fitness')}\n")
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
    """Run Bayesian search, then retrain with the best hyperparameters using the same
    leave-one-out cross-validation loop big_train.py's __main__ runs (see the retrain
    loop below) -- saving/plotting each fold plus an ensemble directly into out_dir.

    Returns a dict with:
        model_type, fold_models, ensemble_model, best_params,
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
        objective=kt.Objective("mean_fitness", direction="max"),
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
    print("Objective (mean_fitness):", best.score)
    print("Hyperparameters:", best.hyperparameters.values)
    print("Final val_loss:", best.metrics.get_last_value("mean_val_loss"))
    print("Final val_loss_penalized:", best.metrics.get_last_value("mean_val_loss_penalized"))
    print("Final train loss:", best.metrics.get_last_value("mean_train_loss"))

    _save_best_trial_details(best, t_elapsed, out_dir)

    # Retrain with the best hyperparameters using the same leave-one-out cross-validation
    # loop big_train.py's __main__ runs -- one model per held-out panel in CV_PANELS, not
    # a single fixed split. mean_fitness/mean_val_loss_penalized during search were
    # averages over these same 4 folds, so this is what actually reflects that evaluation
    # (a single retrain on one fixed split would be a different, unrelated model).
    best_params = best.hyperparameters.values
    seed = int(np.random.randint(0, 10000))

    final_params = {
        "input_size": N_FEAT,
        "n_features": N_FEAT,
        "latent_dim": best_params["latent_dim"],
        "k_sparse": resolve_k_sparse(best_params["k_sparse_frac"], best_params["latent_dim"]),
        "drop_rate": best_params["drop_rate"],
    }
    if model_type == "CNN_AE":
        final_params["filters_bench"] = best_params["filters_bench"]
        final_params["filters_path"] = best_params["filters_path"]
        final_params["n_channels"] = 2  # matches MyTuner.benchmark / build_model above

    rec_train_loss_list, lat_train_loss_list = [], []
    rec_val_loss_list, lat_val_loss_list = [], []
    fold_final_losses = []
    fold_entries = []  # (panel_held_out, model, norm_stats) -- for the ensemble
    last_ds_dict, last_States_dict = None, None
    model_info = None

    for panel in CV_PANELS:
        train_ds_names = [p for p in CV_PANELS if p != panel]
        val_ds_names = [panel]
        tf.random.set_seed(seed)

        (model, history, final_loss, _, _, _, _, _, model_info, ds_dict, _, _, States_dict,
         *_rest) = model_train_features(
            net_type=model_type,
            include_benchmark=(model_type == "CNN_AE"),
            params=final_params,
            path_i=path_i,
            frequency_i=freq,
            train_ds_names=train_ds_names,
            val_ds_names=val_ds_names,
            results_dir=out_dir,
            epochs=EPOCHS_PER_FOLD,
            base_batch_size=best_params["batch_size"],
            loss_weights={"reconstruction": 1.0, "sHI": 2.0},
            seed=seed,
            filepath=f"Model_val_{panel}.keras",
        )
        norm_stats = _rest[-1]

        plot_sHI_cv_fold(model, ds_dict, States_dict, CV_PANELS, panel,
                          save_path=os.path.join(out_dir, f"sHI_fold_val_{panel}.png"))
        plot_reconstruction_cv_fold(model, ds_dict, States_dict, CV_PANELS, panel, 0,
                                     save_path=os.path.join(out_dir, f"reconstruction_fold_val_{panel}.png"))

        rec_train_loss_list.append(history.history['reconstruction_loss'])
        lat_train_loss_list.append(history.history['sHI_loss'])
        rec_val_loss_list.append(history.history['val_reconstruction_loss'])
        lat_val_loss_list.append(history.history['val_sHI_loss'])
        fold_final_losses.append(final_loss)

        fold_entries.append((panel, model, norm_stats))
        last_ds_dict, last_States_dict = ds_dict, States_dict

    _append_to_model_database(model_info, results_dir=db_dir or MODEL_DB_DIR)

    # Learning curves: reconstruction loss (top row) + latent loss (bottom row), one
    # column per panel -- identical layout to big_train.py's __main__.
    avg_final_loss = float(np.mean(fold_final_losses))
    plt.figure(figsize=(16, 6))
    for i, panel in enumerate(CV_PANELS):
        plt.subplot(2, 4, i + 1)
        plt.plot(rec_train_loss_list[i], label='Train')
        plt.plot(rec_val_loss_list[i], label='Val')
        plt.title(f'Recon Loss - Panel {panel}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    for i, panel in enumerate(CV_PANELS):
        plt.subplot(2, 4, i + 5)
        plt.plot(lat_train_loss_list[i], label='Train')
        plt.plot(lat_val_loss_list[i], label='Val')
        plt.title(f'Latent Loss - Panel {panel}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"learning_curves_{model_type}_{avg_final_loss:.4f}.png"))
    plt.close()

    # Ensemble of the 4 cross-validation folds -- same as big_train.py's __main__.
    ensemble_model = build_ensemble_ae(fold_entries)
    ensemble_model.save(os.path.join(out_dir, "ensemble_model.keras"))
    _, _, ref_norm_stats = fold_entries[0]
    plot_ensemble_sHI(
        ensemble_model, last_ds_dict, ref_norm_stats, last_States_dict, CV_PANELS,
        save_path=os.path.join(out_dir, "ensemble_sHI.png"),
    )

    train_h, val_h, fitness_h = _collect_loss_histories(tuner)
    # fitness is maximized (mean_fitness, direction="max"), unlike val_h/train_h above.
    running_best = np.maximum.accumulate(fitness_h) if fitness_h else np.array([])
    df = _build_trials_dataframe(tuner)
    print(df.head(10))

    return {
        "model_type": model_type,
        "fold_models": fold_entries,
        "ensemble_model": ensemble_model,
        "best_params": best_params,
        "train_h": train_h,
        "val_h": val_h,
        "fitness_h": fitness_h,
        "running_best": running_best,
        "df": df,
        "out_dir": out_dir,
    }


# ─── Plotting (works for 1 or N models — single-mode is just N=1) ─────────────
def _min_max_normalize(values):
    # Unlike dividing by the mean, this has no sign-flip / near-zero blow-up risk --
    # val_h (mean_val_loss_penalized) is unbounded and can be negative or near-zero
    # (see monotonicity_loss), which made mean-based normalization unstable.
    x = np.asarray(values, dtype=float)
    lo, hi = x.min(), x.max()
    if hi <= lo:  # all trials had the same value (e.g. a single trial) -- avoid /0
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def plot_optimization_progress(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(_min_max_normalize(r["fitness_h"]), label="Fitness (min-max)")
        ax.plot(_min_max_normalize(r["val_h"]), label="Validation Loss (min-max)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Min-Max Normalized [0, 1]")
        ax.set_title(f"Fitness & Validation Loss per Trial — {r['model_type']}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)


def plot_running_best(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(r["running_best"], "k--", label="Running Best")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best Fitness")
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
    plot_optimization_progress(results, f"{primary_dir}/bayesian_optimization_progress.svg")
    plot_running_best(results,         f"{primary_dir}/bayesian_running_best.svg")
    plot_overfitting(results,          f"{primary_dir}/overfitting_analysis.svg")
    for r in results:
        plot_hp_sensitivity(r, f"{r['out_dir']}/hp_sensitivity.svg")
        plot_hp_coverage(r,    f"{r['out_dir']}/hp_coverage.svg")


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

    folder_name = f"Multi_path_BO_fixed"
    
    for PATH_I in range(3,28):
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
        # fold/ensemble training + plotting already happened inside run_bayesian_optimization

    #plt.show()

        # ── cleanup before next path ─────────────────────
        plt.close("all")           # release matplotlib figures
        del results                # drop refs to model, ds_dict, df, etc.
        tf.keras.backend.clear_session()
        gc.collect()
        log_mem(f"after path {PATH_I} cleanup")  # verify it actually drops


if __name__ == "__main__":
    main()