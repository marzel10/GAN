'''
Summarizes the Bayesian-optimized GCN hyperparameters across frequencies, parsed from
the best_trial_details.txt that BO_GCN.py's run_bayesian_optimization writes into each
Bayesian_GCN_freq{N}/ folder.

best_trial_details.txt has one plain-text line per field, e.g.:
    Optimization time 0 days 00:30:38.441542
    Trial ID: 28
    Objective: 23.313475868662096
    Hyperparameters: {'nr_hidden_channels': 0, 'hidden_dim': 8, 'dropout': 0.2, 'batch_size': 32, 'learning_rate': 0.00013626331785482134}
    Final mean_fitness: 2.424689781625961
    Final mean_damage_loss: -2088.8786087036133
    Final mean_train_loss: -3334.312505086263
    Final mean_val_loss: -4915.289154052734

Plots, one per hyperparameter, frequency index on the x-axis (one point per
frequency -- GCN trains one model across all paths jointly, so there's no per-path
axis the way AE_hyperparameters_summary.py has):
    - nr_hidden_channels
    - hidden_dim
    - dropout
    - batch_size
    - learning_rate
and one per best-trial outcome metric:
    - Objective
    - mean_fitness
    - mean_damage_loss
    - mean_train_loss
    - mean_val_loss

'''
import ast
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
from matplotlib.ticker import MaxNLocator
import pandas as pd

from config import BO_SEARCH_RESULTS_DIR, PROJECT_ROOT

FREQS = range(0, 6)
FOLDERS = [f"Bayesian_GCN_freq{freq}" for freq in FREQS]
OUT_DIR = PROJECT_ROOT / "GCN_hyperparameters_summary_results"

# (key, plot title / y-axis label)
HP_PLOTS = [
    ("nr_hidden_channels", "Optimal nr_hidden_channels"),
    ("hidden_dim", "Optimal hidden_dim"),
    ("dropout", "Optimal dropout"),
    ("batch_size", "Optimal batch_size"),
    ("learning_rate", "Optimal learning_rate"),
]
METRIC_PLOTS = [
    ("Objective", "Best-trial Objective"),
    ("mean_fitness", "Best-trial mean_fitness"),
    ("mean_damage_loss", "Best-trial mean_damage_loss"),
    ("mean_train_loss", "Best-trial mean_train_loss"),
    ("mean_val_loss", "Best-trial mean_val_loss"),
]


def _parse_best_trial_details(txt_path):
    row = {}
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Optimization time "):
            row["optimization_time"] = line[len("Optimization time "):]
        elif line.startswith("Trial ID:"):
            row["trial_id"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Objective:"):
            row["Objective"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Hyperparameters:"):
            hp = ast.literal_eval(line.split(":", 1)[1].strip())
            row.update(hp)
        elif line.startswith("Final "):
            key, value = line[len("Final "):].split(":", 1)
            row[key.strip()] = float(value.strip())
    return row


def load_all_trials(folders=FOLDERS, root=BO_SEARCH_RESULTS_DIR):
    '''One row per frequency folder's best_trial_details.txt, tagged with its frequency_index.'''
    rows = []
    for freq, folder in zip(FREQS, folders):
        txt_path = root / folder / "best_trial_details.txt"
        if not txt_path.exists():
            print(f"[{folder}] no best_trial_details.txt, skipping")
            continue
        row = _parse_best_trial_details(txt_path)
        row["frequency_index"] = freq
        row["source_folder"] = folder
        rows.append(row)
        print(f"[{folder}] loaded best trial (trial_id={row.get('trial_id')})")

    if not rows:
        raise FileNotFoundError(f"No best_trial_details.txt found in any of {folders} (looked under {root})")

    return pd.DataFrame(rows)


def _plot_vs_frequency(df, columns, out_dir, file_prefix):
    out_dir.mkdir(parents=True, exist_ok=True)

    for col, title in columns:
        if col not in df.columns:
            print(f"[{col}] column missing from every loaded trial, skipping plot")
            continue
        sub = df.dropna(subset=[col, "frequency_index"]).sort_values("frequency_index")
        if sub.empty:
            print(f"[{col}] no non-null rows, skipping plot")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub["frequency_index"], sub[col], "o-", color="tab:blue")

        ax.set_xlabel("Frequency index")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Frequency")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        fig.tight_layout()
        save_path = out_dir / f"{file_prefix}_{col}.svg"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_hp_vs_freq(df, out_dir=OUT_DIR):
    _plot_vs_frequency(df, HP_PLOTS, out_dir, "GCN_hp_summary")


def plot_metrics_vs_freq(df, out_dir=OUT_DIR):
    _plot_vs_frequency(df, METRIC_PLOTS, out_dir, "GCN_metric_summary")

def make_table_summary(df, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "GCN_hyperparameters_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

def main():
    df = load_all_trials()
    print(f"Loaded {len(df)} row(s) total from {df['source_folder'].nunique()} folder(s)")
    plot_hp_vs_freq(df)
    plot_metrics_vs_freq(df)
    make_table_summary(df)


if __name__ == "__main__":
    main()
