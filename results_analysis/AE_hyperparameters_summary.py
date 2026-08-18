'''
Summarizes the Bayesian-optimized AE hyperparameters across every path, aggregated
over the per-frequency model_database.xlsx files that BO_AE.py's
_append_to_model_database writes into each Multi_path_BO_fixed_freq{N}/ folder (one
row per path -- the LAST cross-validation fold retrained in run_bayesian_optimization,
since model_info is overwritten each fold and only saved once per path).

Plots, one per hyperparameter, path number on the x-axis, one colored series per
frequency:
    - k_sparse   (the resolved absolute count -- see BO_features.resolve_k_sparse --
                  not the k_sparse_frac hyperparameter the search actually samples)
    - filters_bench
    - filters_path
    - batch_size

Folders/columns that don't exist yet (frequency not swept, or a fc_AE run that has no
filters_bench/filters_path) are skipped rather than erroring.
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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from config import PROJECT_ROOT, TEST_RUN_DIR, CUSTOM_PALETTE

FOLDERS = [f"Multi_path_BO_fixed_freq{freq}" for freq in range(0, 6)]
OUT_DIR = TEST_RUN_DIR / "AE_hyperparameters_summary_results"

_MARKERS = ["o", "s", "^", "D"]


def _palette_style(i):
    color = CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)]
    marker = _MARKERS[(i // len(CUSTOM_PALETTE)) % len(_MARKERS)]
    return color, marker

# (column in model_database.xlsx, plot title / y-axis label)
HP_PLOTS = [
    ("k_sparse", "Optimal latent size (k-sparse)"),
    ("filters_bench", "Optimal benchmark filter count"),
    ("filters_path", "Optimal path filter count"),
    ("batch_size", "Optimal batch size"),
]


def load_all_databases(folders=FOLDERS, root=TEST_RUN_DIR):
    '''Concatenates every folder's model_database.xlsx into one DataFrame, tagging
    each row with its source folder'''
    frames = []
    for folder in folders:
        xlsx_path = root / folder / "model_database.xlsx"
        if not xlsx_path.exists():
            print(f"[{folder}] no model_database.xlsx, skipping")
            continue
        df = pd.read_excel(xlsx_path)
        df["source_folder"] = folder
        frames.append(df)
        print(f"[{folder}] loaded {len(df)} row(s)")

    if not frames:
        raise FileNotFoundError(f"No model_database.xlsx found in any of {folders} (looked under {root})")

    return pd.concat(frames, ignore_index=True)


def plot_hp_vs_path(df, out_dir=OUT_DIR):
    '''One figure per hyperparameter: value vs path_index, one series per frequency
    (colors = each row's own frequency_index, not the folder it came from -- they
    should always agree, but frequency_index is what BO_features.py actually swept).'''
    out_dir.mkdir(parents=True, exist_ok=True)

    if "frequency_index" not in df.columns or "path_index" not in df.columns:
        raise KeyError("model_database.xlsx is missing 'frequency_index' or 'path_index' -- "
                        "unexpected schema, can't plot vs path/frequency.")

    freqs = sorted(df["frequency_index"].dropna().unique())

    for col, title in HP_PLOTS:
        if col not in df.columns:
            print(f"[{col}] column missing from every loaded database, skipping plot")
            continue
        sub = df.dropna(subset=[col, "path_index", "frequency_index"])
        if sub.empty:
            print(f"[{col}] no non-null rows, skipping plot")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        any_series = False
        for i, freq in enumerate(freqs):
            freq_rows = sub[sub["frequency_index"] == freq].sort_values("path_index")
            if freq_rows.empty:
                continue
            color, marker = _palette_style(i)
            ax.plot(freq_rows["path_index"], freq_rows[col], marker=marker, linestyle="None",
                    color=color, label=f"freq {int(freq)}")
            any_series = True
        if not any_series:
            plt.close(fig)
            continue

        ax.set_xlabel("Path number")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Path")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(title="Frequency index")
        ax.grid(True)
        fig.tight_layout()
        save_path = out_dir / f"AE_hp_summary_{col}.svg"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")


def main():
    df = load_all_databases()
    print(f"Loaded {len(df)} row(s) total from {df['source_folder'].nunique()} folder(s)")
    plot_hp_vs_path(df)


if __name__ == "__main__":
    main()
