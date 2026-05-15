import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from python_project.big_train import monotonicity_loss, monotonicity_loss_v2
from python_project.fc_AE import build_CNN_variable_block, build_deep_CNN_network, build_deep_fully_connected_network
from python_project.states_check import prepare_datastores
import keras_tuner as kt
import pandas as pd
import seaborn as sns
import os
from functools import partial
from python_project.results_viz import plot_sHI_vs_RUL, PlotContext
from python_project.big_train import model_train

K_SPARSE_PENALTY_WEIGHT = 0.01


def k_sparse_penalty(k, weight=K_SPARSE_PENALTY_WEIGHT):
    return float(weight) * float(k)


def get_k_sparse_hp(hp):
    return hp.Int("k_sparse", min_value=5, max_value=15, step=1)


def get_batch_size_hp(hp):
    return hp.Int("batch_size", min_value=4, max_value=32, step=4)




class KSparsePenaltyCallback(tf.keras.callbacks.Callback):
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


def build_model(hp, model_type="fc_AE"):
    # Declare batch_size here so KerasTuner includes it in the search space
    # and prints it starting from Trial #1 (even though it's used in run_trial).
   
    if model_type == "fc_AE":
        params = {
            "hidden_layer_size1": hp.Int("hidden_layer_size1", min_value=512, max_value=2048, step=256),
            "hidden_layer_size2": hp.Int("hidden_layer_size2", min_value=256, max_value=1024, step=128),
            "hidden_layer_size3": hp.Int("hidden_layer_size3", min_value=128, max_value=512, step=64),
            "hidden_layer_size4": hp.Int("hidden_layer_size4", min_value=64, max_value=256, step=32),
            "desired_latent_size": 15,
            "drop_rate": hp.Float("drop_rate", min_value=0.0, max_value=0.5, step=0.1),
            "k_sparse": get_k_sparse_hp(hp),
            "input_size": 4000,
            "num_in": 1,
        }
        model = build_deep_fully_connected_network(params)
    elif model_type == "CNN_AE":
        params = {
            # Standardize kernel sizes to 3 and 5
            "kernel_size1": hp.Choice("kernel_size1", values=[50,20,10, 5]),
            "kernel_size2": hp.Choice("kernel_size2", values=[20,10, 5,3]),
            # "kernel_size3": hp.Choice("kernel_size3", values=[3, 5]),
            # # "kernel_size4": hp.Choice("kernel_size4", values=[3, 5]),
            # "drop_rate": hp.Float("drop_rate", min_value=0.0, max_value=0.3, step=0.05),
            #"k_sparse": get_k_sparse_hp(hp),
            # Significantly reduced filter counts to avoid hardware aborts
            "filter1": hp.Choice("filter1", values=[32, 64, 128]),
            "filter2": hp.Choice("filter2", values=[16, 32, 64]),
            # "filter3": hp.Choice("filter3", values=[128, 256, 512]),
            # "filter4": hp.Choice("filter4", values=[16, 32, 64]), # Latent/Bottleneck area
            "pool_size1": hp.Choice("pool_size1", values=[20,10,5]),
            "pool_size2": hp.Choice("pool_size2", values=[10,5,2]),
            "n_blocks": 2,
        }
        model = build_CNN_variable_block(params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    Vlearning_rate = 0.001
    Voptimizer = tf.keras.optimizers.Adam(learning_rate=Vlearning_rate)

    # Output naming differs between models:
    # - FC AE: reconstruction output layer is "fc_output_1"
    # - CNN_AE AE: reconstruction output layer is "final_1" (after crop/pad)
    recon_name = "final_1" if model_type == "CNN_AE" else "fc_output_1"
    Vloss_weights = {recon_name: 0.5, "fc_latent_1": 1.0}
    Vloss_fun = {recon_name: "mse", "fc_latent_1": "mse"}
    model.compile(
        optimizer=Voptimizer,
        loss=Vloss_fun,
        loss_weights=Vloss_weights,
    )
    return model

class MyTuner(kt.BayesianOptimization):
    def __init__(self, *args, model_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = get_batch_size_hp(hp)
        k_sparse = get_k_sparse_hp(hp)
        

        train_ds, val_ds, _, _, _, _,_ = prepare_datastores(
            path_i=0,
            freq_i=3,
            base_batch_size=bs,
            test_batch_size=1,
            train_ds_names=["103", "104", "105"],
            val_ds_names=["109"],
            test_ds_names=["123_1", "123_31", "123_41", "123_43","123_2", "123_32", "123_42", "123_44"],
            include_benchmark=[True if self.model_type == "CNN_AE" else False][0]
        )

        kwargs["x"] = train_ds
        kwargs["validation_data"] = val_ds
        kwargs["epochs"] = 1  # epochs per trial
        kwargs.setdefault("verbose", 1)

        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(KSparsePenaltyCallback(k_sparse=k_sparse))
        kwargs["callbacks"] = callbacks

        return super().run_trial(trial, *args, **kwargs)

def run_bayesian_optimization(
        model_type="fc_AE",
        max_trials=30,
    ):
      
    # search() needs no x/epochs args — Hyperband manages those internally
    t_start = pd.Timestamp.now()
    if model_type == "fc_AE":
        model_building_function = build_model
    elif model_type == "CNN_AE":
        model_building_function = partial(build_model, model_type="CNN_AE")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    tuner = MyTuner(
        model_building_function,
        objective=kt.Objective("val_loss_penalized", direction="min"),
        max_trials=30,
        directory="tuner_dir",
        project_name="ae",
        overwrite=True,  # overwrite previous results for clean runs
        model_type=model_type,
    )
    tuner.search()  # no args needed
    t_end = pd.Timestamp.now()
    print(f"Bayesian optimization completed in {t_end - t_start}")
    # create saving directory
    date = pd.Timestamp.now().strftime("%Y_%m_%d-%H_%M_%S")
    dir = f"results/Bayesian_{date}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    best = tuner.oracle.get_best_trials(1)[0]

    print("Trial ID:", best.trial_id)
    print("Objective (val_loss_penalized):", best.score)
    print("Hyperparameters:", best.hyperparameters.values)
    print("Final val_loss:", best.metrics.get_last_value("val_loss"))
    print("Final val_loss_penalized:", best.metrics.get_last_value("val_loss_penalized"))
    print("Final train loss:", best.metrics.get_last_value("loss"))
    print("Val loss history:", best.metrics.get_history("val_loss"))
    print("Val loss penalized history:", best.metrics.get_history("val_loss_penalized"))

    with open(f"{dir}/best_trial_details.txt", "w") as f:
        f.write(f"Optimization time {t_end - t_start}\n")
        f.write(f"Trial ID: {best.trial_id}\n")
        f.write(f"Objective (val_loss_penalized): {best.score}\n")
        f.write(f"Hyperparameters: {best.hyperparameters.values}\n")
        f.write(f"Final val_loss: {best.metrics.get_last_value('val_loss')}\n")
        f.write(f"Final val_loss_penalized: {best.metrics.get_last_value('val_loss_penalized')}\n")
        f.write(f"Final train_loss: {best.metrics.get_last_value('loss')}\n")
        f.write(f"Val loss history: {best.metrics.get_history('val_loss')}\n")
        f.write(f"Val loss penalized history: {best.metrics.get_history('val_loss_penalized')}\n")

    # Train and save the best model
    best_params = best.hyperparameters.values
    #generate random seed for reproducibility
    seed = np.random.randint(0, 10000)
    tf.random.set_seed(seed)
    best_model, _, _, model_info, _, _, _, _, _, _, _ = model_train(model_type, params=best_params, seed=seed, n_blocks=2)
    # 1. Capture model summary as a string for the text file
    results_dir = "results_hype_tweaked/"
    stream = io.StringIO()
    best_model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()

    # 2. Get the number of trainable parameters
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in best_model.trainable_weights])



    # --- SAVE TO EXCEL DATABASE ---
    excel_path = f"{results_dir}model_database.xlsx"

    # Create a DataFrame from the single run
    df_new = pd.DataFrame([model_info])


    if not os.path.exists(excel_path):
        # If file doesn't exist, create it
        df_new.to_excel(excel_path, index=False)
        print(f"Created new database: {excel_path}")
    else:
        # If exists, append the new row
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Load existing data to find the next empty row
            try:
                existing_df = pd.read_excel(excel_path)
                df_new.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
            except Exception as e:
                # Fallback if file is corrupted/empty
                df_new.to_excel(excel_path, index=False)
        print(f"Updated database at {excel_path}")

    val_h = []
    train_h = []
    def _scalar(v):
        # KerasTuner returns MetricObservation(value=[...], step=...).
        # Value can be a list-like (often length-1), numpy scalar/array, or tf.Tensor.
        if hasattr(v, "value"):
            v = v.value
        if isinstance(v, tf.Tensor):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            v = v.reshape(-1)
            return float(v[0])
        if isinstance(v, (list, tuple)):
            return float(v[0])
        return float(v)

    for t in tuner.oracle.trials.values():

        if t.score is None:  # skip incomplete trials
            continue
        val_obs   = [_scalar(obs.value) for obs in t.metrics.get_history("val_loss_penalized")]
        train_obs = [_scalar(obs.value) for obs in t.metrics.get_history("loss_penalized")]
        if val_obs and train_obs:
            val_h.append(min(val_obs))
            train_h.append(min(train_obs))

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(val_h,   label="Validation Loss")
    ax1.plot(train_h, label="Training Loss")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Best Loss")
    ax1.set_title("Best Loss per Trial (Bayesian Optimization Progress)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{dir}/bayesian_optimization_progress.png")


    running_best = np.minimum.accumulate(val_h)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(running_best, "k--", label="Running Best")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Best Validation Loss")
    ax2.set_title("Running Best Validation Loss Across Trials")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{dir}/bayesian_running_best.png")

    # ── 1. Collect all trials into a DataFrame ──────────────────────────────────
    records = []
    for tid, t in tuner.oracle.trials.items():
        if t.score is None:
            continue
        row = {"trial_id": tid, "objective": t.score}
        row.update(t.hyperparameters.values)

        val_hist = [_scalar(obs) for obs in t.metrics.get_history("val_loss")]
        train_hist = [_scalar(obs) for obs in t.metrics.get_history("loss")]
        val_pen_hist = [_scalar(obs) for obs in t.metrics.get_history("val_loss_penalized")]
        train_pen_hist = [_scalar(obs) for obs in t.metrics.get_history("loss_penalized")]

        row["final_val_loss"] = val_hist[-1] if val_hist else None
        row["final_train_loss"] = train_hist[-1] if train_hist else None
        row["final_val_loss_penalized"] = val_pen_hist[-1] if val_pen_hist else None
        row["final_train_loss_penalized"] = train_pen_hist[-1] if train_pen_hist else None
        
        row["overfit_gap"]      = (
            row["final_val_loss"] - row["final_train_loss"]
            if (row["final_val_loss"] is not None and row["final_train_loss"] is not None) else None
        )
        records.append(row)

    df = pd.DataFrame(records).sort_values("objective")
    print(df.head(10))  # top-10 trials

    # ── 2. HP sensitivity: scatter each HP vs val_loss ──────────────────────────
    if model_type == "fc_AE":
        hp_cols = [
            "hidden_layer_size1", "hidden_layer_size2", "hidden_layer_size3",
            "hidden_layer_size4",  "drop_rate",
            "k_sparse", "batch_size"
        ]
    elif model_type == "CNN_AE":
        # hp_cols = [
        #     "kernel_size1", "kernel_size2", "kernel_size3", "kernel_size4",
        #     "drop_rate", "k_sparse", "filter1", "filter2", "filter3", "filter4", "batch_size"
        # ]
        hp_cols = ["kernel_size1", "kernel_size2","pool_size1","pool_size2","filter1", "filter2", "batch_size"]

    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes3.flat, hp_cols):
        ax.scatter(df[col], df["objective"], alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(col)
        ax.set_ylabel("objective")
        ax.set_title(f"{col} vs objective")
    fig3.suptitle("Hyperparameter Sensitivity", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(f"{dir}/hp_sensitivity.png")

    # ── 3. HP search coverage histograms ────────────────────────────────────────
    fig4, axes4 = plt.subplots(2, 4, figsize=(18, 6))
    for ax, col in zip(axes4.flat, hp_cols):
        ax.hist(df[col].dropna(), bins=10, edgecolor="k")
        ax.set_title(col)
    fig4.suptitle("Sampled HP Distribution (search coverage)", fontsize=14)
    fig4.tight_layout()
    fig4.savefig(f"{dir}/hp_coverage.png")

    # ── 4. Overfitting analysis ──────────────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    sc = ax5.scatter(df["final_train_loss"], df["final_val_loss"],
            c=df["overfit_gap"], cmap="RdYlGn_r", edgecolors="k", s=60)
    fig5.colorbar(sc, ax=ax5, label="overfit gap (val - train)")
    ax5.plot([df["final_train_loss"].min(), df["final_train_loss"].max()],
            [df["final_train_loss"].min(), df["final_train_loss"].max()],
            "k--", lw=1, label="no overfit line")
    ax5.set_xlabel("Final train loss")
    ax5.set_ylabel("Final val loss")
    ax5.set_title("Overfitting Analysis Across Trials")
    ax5.legend()
    fig5.tight_layout()
    fig5.savefig(f"{dir}/overfitting_analysis.png")

    return best_model, best_params, train_h, val_h, running_best, df, dir


'''
model_type = "CNN_AE"  # or "CNN_AE"   
# search() needs no x/epochs args — Hyperband manages those internally
t_start = pd.Timestamp.now()
if model_type == "fc_AE":
    model_building_function = build_model
elif model_type == "CNN_AE":
    model_building_function = partial(build_model, model_type="CNN_AE")
else:
    raise ValueError(f"Unknown model_type: {model_type}")

tuner = MyTuner(
    model_building_function,
    objective=kt.Objective("val_loss_penalized", direction="min"),
    max_trials=2,
    directory="tuner_dir",
    project_name="ae",
    overwrite=True,  # overwrite previous results for clean runs
    model_type=model_type,
)
tuner.search()  # no args needed
t_end = pd.Timestamp.now()
print(f"Bayesian optimization completed in {t_end - t_start}")
# create saving directory
date = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
dir = f"results/Bayesian_{date}"
if not os.path.exists(dir):
    os.makedirs(dir)

best = tuner.oracle.get_best_trials(1)[0]

print("Trial ID:", best.trial_id)
print("Objective (val_loss_penalized):", best.score)
print("Hyperparameters:", best.hyperparameters.values)
print("Final val_loss:", best.metrics.get_last_value("val_loss"))
print("Final val_loss_penalized:", best.metrics.get_last_value("val_loss_penalized"))
print("Final train loss:", best.metrics.get_last_value("loss"))
print("Val loss history:", best.metrics.get_history("val_loss"))
print("Val loss penalized history:", best.metrics.get_history("val_loss_penalized"))

with open(f"{dir}/best_trial_details.txt", "w") as f:
    f.write(f"Optimization time {t_end - t_start}\n")
    f.write(f"Trial ID: {best.trial_id}\n")
    f.write(f"Objective (val_loss_penalized): {best.score}\n")
    f.write(f"Hyperparameters: {best.hyperparameters.values}\n")
    f.write(f"Final val_loss: {best.metrics.get_last_value('val_loss')}\n")
    f.write(f"Final val_loss_penalized: {best.metrics.get_last_value('val_loss_penalized')}\n")
    f.write(f"Final train_loss: {best.metrics.get_last_value('loss')}\n")
    f.write(f"Val loss history: {best.metrics.get_history('val_loss')}\n")
    f.write(f"Val loss penalized history: {best.metrics.get_history('val_loss_penalized')}\n")

# Train and save the best model
best_params = best.hyperparameters.values
#generate random seed for reproducibility
seed = np.random.randint(0, 10000)
tf.random.set_seed(seed)
best_model, _, _, model_info, _, _, _, _, _, _, _ = model_train(model_type, params=best_params, seed=seed, epochs=1)
# 1. Capture model summary as a string for the text file
results_dir = "results_hype_tweaked/"
stream = io.StringIO()
best_model.summary(print_fn=lambda x: stream.write(x + '\n'))
summary_str = stream.getvalue()

# 2. Get the number of trainable parameters
trainable_count = np.sum([tf.keras.backend.count_params(w) for w in best_model.trainable_weights])



# --- SAVE TO EXCEL DATABASE ---
excel_path = f"{results_dir}model_database.xlsx"

# Create a DataFrame from the single run
df_new = pd.DataFrame([model_info])

if not os.path.exists(excel_path):
    # If file doesn't exist, create it
    df_new.to_excel(excel_path, index=False)
    print(f"Created new database: {excel_path}")
else:
    # If exists, append the new row
    with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        # Load existing data to find the next empty row
        try:
            existing_df = pd.read_excel(excel_path)
            df_new.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
        except Exception as e:
            # Fallback if file is corrupted/empty
            df_new.to_excel(excel_path, index=False)
    print(f"Updated database at {excel_path}")

val_h = []
train_h = []
def _scalar(v):
    # KerasTuner returns MetricObservation(value=[...], step=...).
    # Value can be a list-like (often length-1), numpy scalar/array, or tf.Tensor.
    if hasattr(v, "value"):
        v = v.value
    if isinstance(v, tf.Tensor):
        v = v.numpy()
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)
        return float(v[0])
    if isinstance(v, (list, tuple)):
        return float(v[0])
    return float(v)

for t in tuner.oracle.trials.values():

    if t.score is None:  # skip incomplete trials
        continue
    val_obs   = [_scalar(obs.value) for obs in t.metrics.get_history("val_loss_penalized")]
    train_obs = [_scalar(obs.value) for obs in t.metrics.get_history("loss_penalized")]
    if val_obs and train_obs:
        val_h.append(min(val_obs))
        train_h.append(min(train_obs))
'''
double_mode = True
model_type = "fc_AE" #"CNN_AE"
best_model_fc, best_params_fc, train_h_fc, val_h_fc, running_best_fc, df_fc, results_dir_fc = run_bayesian_optimization(
    model_type=model_type, 
    max_trials=50 # Set your trials here
)

model_type = "CNN_AE"
best_model_cnn, best_params_cnn, train_h_cnn, val_h_cnn, running_best_cnn, df_cnn, results_dir_cnn = run_bayesian_optimization(
    model_type=model_type, 
    max_trials=50 # Set your trials here
)

# Two subplots side by side for FC AE and CNN AE
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
ax1[0].plot(val_h_fc,   label="Validation Loss")
ax1[0].plot(train_h_fc, label="Training Loss")
ax1[0].set_xlabel("Trial")
ax1[0].set_ylabel("Best Loss")
ax1[0].set_title("Best Loss per Trial (Bayesian Optimization Progress)")
ax1[0].legend()
ax1[1].plot(val_h_cnn,   label="Validation Loss")
ax1[1].plot(train_h_cnn, label="Training Loss")
ax1[1].set_xlabel("Trial")
ax1[1].set_ylabel("Best Loss")
ax1[1].set_title("Best Loss per Trial (Bayesian Optimization Progress)")
ax1[1].legend()
fig1.tight_layout()
fig1.savefig(f"{results_dir_fc}/bayesian_optimization_progress.png")


running_best = np.minimum.accumulate(val_h_fc)
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
ax2[0].plot(running_best, "k--", label="Running Best")
ax2[0].set_xlabel("Trial")
ax2[0].set_ylabel("Best Validation Loss")
ax2[0].set_title("Running Best Validation Loss Across Trials")
ax2[0].legend()
running_best = np.minimum.accumulate(val_h_cnn)
ax2[1].plot(running_best, "k--", label="Running Best")
ax2[1].set_xlabel("Trial")
ax2[1].set_ylabel("Best Validation Loss")
ax2[1].set_title("Running Best Validation Loss Across Trials")
ax2[1].legend()
fig2.tight_layout()
fig2.savefig(f"{results_dir_fc}/bayesian_running_best.png")

#print(df_fc.head(10))  # top-10 trials

# ── 2. HP sensitivity: scatter each HP vs val_loss ──────────────────────────
if not double_mode:
    if model_type == "fc_AE":
        hp_cols = [
            "hidden_layer_size1", "hidden_layer_size2", "hidden_layer_size3",
            "hidden_layer_size4",  "drop_rate",
            "k_sparse", "batch_size"
        ]
    elif model_type == "CNN_AE":
        # hp_cols = [
        #     "kernel_size1", "kernel_size2", "kernel_size3", "kernel_size4",
        #     "drop_rate", "k_sparse", "filter1", "filter2", "filter3", "filter4", "batch_size"
        # ]
        hp_cols = [
            "kernel_size1", "kernel_size2","pool_size1","pool_size2",
            "filter1", "filter2", "batch_size"
        ]
else:
    hp_cols_fc = [
        "hidden_layer_size1", "hidden_layer_size2", "hidden_layer_size3",
        "hidden_layer_size4",  "drop_rate",
        "k_sparse", "batch_size"
    ]
    # hp_cols_cnn = [
    #     "kernel_size1", "kernel_size2", "kernel_size3", "kernel_size4",
    #     "drop_rate", "k_sparse", "filter1", "filter2", "filter3", "filter4", "batch_size"
    # ]
    hp_cols_cnn = [
        "kernel_size1", "kernel_size2","pool_size1","pool_size2",
        "filter1", "filter2", "batch_size"
    ]


fig3, axes3 = plt.subplots(2, 4, figsize=(20, 8))
for ax, col in zip(axes3.flat, hp_cols_fc):
    ax.scatter(df_fc[col], df_fc["objective"], alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(col)
    ax.set_ylabel("objective")
    ax.set_title(f"{col} vs objective")
fig3.suptitle("Hyperparameter Sensitivity", fontsize=14)
fig3.tight_layout()
fig3.savefig(f"{results_dir_fc}/hp_sensitivity.png")

fig3_cnn, axes3_cnn = plt.subplots(2, 5, figsize=(25, 8))
for ax, col in zip(axes3_cnn.flat, hp_cols_cnn):
    ax.scatter(df_cnn[col], df_cnn["objective"], alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(col)
    ax.set_ylabel("objective")
    ax.set_title(f"{col} vs objective")
fig3_cnn.suptitle("Hyperparameter Sensitivity", fontsize=14)
fig3_cnn.tight_layout()
fig3_cnn.savefig(f"{results_dir_cnn}/hp_sensitivity.png")


# ── 3. HP search coverage histograms ────────────────────────────────────────
fig4, axes4 = plt.subplots(2, 4, figsize=(20, 8))
for ax, col in zip(axes4.flat, hp_cols_fc):
    ax.hist(df_fc[col].dropna(), bins=10, edgecolor="k")
    ax.set_title(col)
fig4.suptitle("Sampled HP Distribution (search coverage)", fontsize=14)
fig4.tight_layout()
fig4.savefig(f"{results_dir_fc}/hp_coverage.png")

fig4_cnn, axes4_cnn = plt.subplots(2, 5, figsize=(25, 8))
for ax, col in zip(axes4_cnn.flat, hp_cols_cnn):
    ax.hist(df_cnn[col].dropna(), bins=10, edgecolor="k")
    ax.set_title(col)
fig4_cnn.suptitle("Sampled HP Distribution (search coverage)", fontsize=14)
fig4_cnn.tight_layout()
fig4_cnn.savefig(f"{results_dir_cnn}/hp_coverage.png")

# ── 4. Overfitting analysis ──────────────────────────────────────────────────
fig5, ax5 = plt.subplots(1, 2, figsize=(14, 6))
sc = ax5[0].scatter(df_fc["final_train_loss"], df_fc["final_val_loss"],
        c=df_fc["overfit_gap"], cmap="RdYlGn_r", edgecolors="k", s=60)
fig5.colorbar(sc, ax=ax5[0], label="overfit gap (val - train)")
ax5[0].plot([df_fc["final_train_loss"].min(), df_fc["final_train_loss"].max()],
        [df_fc["final_train_loss"].min(), df_fc["final_train_loss"].max()],
        "k--", lw=1, label="no overfit line")
ax5[0].set_xlabel("Final train loss")
ax5[0].set_ylabel("Final val loss")
ax5[0].set_title("Overfitting Analysis Across Trials")
ax5[0].legend()
sc = ax5[1].scatter(df_cnn["final_train_loss"], df_cnn["final_val_loss"],
        c=df_cnn["overfit_gap"], cmap="RdYlGn_r", edgecolors="k", s=60)
fig5.colorbar(sc, ax=ax5[1], label="overfit gap (val - train)")
ax5[1].plot([df_cnn["final_train_loss"].min(), df_cnn["final_train_loss"].max()],
        [df_cnn["final_train_loss"].min(), df_cnn["final_train_loss"].max()],
        "k--", lw=1, label="no overfit line")       
ax5[1].set_xlabel("Final train loss")
ax5[1].set_ylabel("Final val loss")
ax5[1].set_title("Overfitting Analysis Across Trials")
ax5[1].legend() 
fig5.tight_layout()
fig5.savefig(f"{results_dir_fc}/overfitting_analysis.png")

''''
# Train model with best HPs and save results
best_hp = best_params
best_model = build_model(best_hp, model_type=model_type)
train_ds, val_ds, test_ds, ds_dict, target_dict, RUL_dict, States_dict = prepare_datastores(
    path_i=0,
    freq_i=3,
    base_batch_size=best_hp.get("batch_size"),
    test_batch_size=1,
    train_ds_names=["103", "104", "105", "109"],
    val_ds_names=["123_1", "123_31", "123_41", "123_43"],
    test_ds_names=["123_2", "123_32", "123_42", "123_44"],
    include_benchmark=(model_type == "CNN_AE"),
)

ctx = PlotContext(
    ds_dict=ds_dict,
    target_dict=target_dict,
    RUL_dict=RUL_dict,
    train_ds_names=["103", "104", "105", "109"],
    val_ds_names=["123_1", "123_31", "123_41", "123_43"],
    test_ds_names=["123_2", "123_32", "123_42", "123_44"],
    States_dict=States_dict
)
history = best_model.fit(train_ds, epochs=50, validation_data=val_ds)
final_val_loss = history.history['val_loss'][-1]
best_model.save(f"{dir}/best_model_val_loss_{final_val_loss:.4f}.keras")
print(f"Best model saved with final val_loss: {final_val_loss:.4f}")
'''

train_ds, val_ds, test_ds, ds_dict, target_dict, RUL_dict, States_dict = prepare_datastores(
    path_i=0,
    freq_i=3,
    base_batch_size=best_params_fc.get("batch_size"),
    test_batch_size=1,
    train_ds_names=["103", "104", "105"],
    val_ds_names=["109"],
    test_ds_names=["123_1", "123_31", "123_41", "123_43","123_2", "123_32", "123_42", "123_44"],
    include_benchmark=False
    #include_benchmark=(model_type == "CNN_AE"),
)
ctx = PlotContext(
    ds_dict=ds_dict,
    target_dict=target_dict,
    RUL_dict=RUL_dict,
    train_ds_names=["103", "104", "105"],
    validation_ds_names=["109"],
    test_ds_names=["123_1", "123_31", "123_41", "123_43","123_2", "123_32", "123_42", "123_44"],
    States_dict=States_dict
)
# make predictions 
state_idx = 0
plot_sHI_vs_RUL(best_model_fc, ctx, state_idx, "train", save=True, dir=results_dir_fc, plot_name="train_val_sHI_vs_RUL", show=False)
plot_sHI_vs_RUL(best_model_fc, ctx, state_idx, "validation", save=True, dir=results_dir_fc, plot_name="val_sHI_vs_RUL", show=False)

train_ds_cnn, val_ds_cnn, test_ds_cnn, ds_dict_cnn, target_dict_cnn, RUL_dict_cnn, States_dict_cnn = prepare_datastores(
    path_i=0,
    freq_i=3,
    base_batch_size=best_params_fc.get("batch_size"),
    test_batch_size=1,
    train_ds_names=["103", "104", "105", "109"],
    val_ds_names=["109"],
    test_ds_names=["123_1", "123_31", "123_41", "123_43","123_2", "123_32", "123_42", "123_44"],
    include_benchmark=True
    #include_benchmark=(model_type == "CNN_AE"),
)
ctx_cnn = PlotContext(
    ds_dict=ds_dict_cnn,
    target_dict=target_dict_cnn,
    RUL_dict=RUL_dict_cnn,
    train_ds_names=["103", "104", "105", "109"],
    validation_ds_names=["109"],
    test_ds_names=["123_1", "123_31", "123_41", "123_43","123_2", "123_32", "123_42", "123_44"],
    States_dict=States_dict_cnn
)
plot_sHI_vs_RUL(best_model_cnn, ctx_cnn, state_idx, "train", save=True, dir=results_dir_cnn, plot_name="train_val_sHI_vs_RUL_bench", show=False)
plot_sHI_vs_RUL(best_model_cnn, ctx_cnn, state_idx, "validation", save=True, dir=results_dir_cnn, plot_name="val_sHI_vs_RUL_bench", show=False) 

plt.show()

