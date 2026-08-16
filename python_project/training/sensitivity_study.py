import multiprocessing
import time

import numpy as np

from BO_GCN import run_bayesian_optimization
from config import PROJECT_ROOT, BETA_CONSTANT, FREQUENCY_MAPPING, TYPES, FREQ_FOR_BETA_SWEEP, BETAS, OPTIMIZED_TYPES, OPTIMIZE_RAW
from graph_performance import main as graph_performance_main
from graph_performance_beta_sweep import main as graph_performance_beta_sweep_main
from Fitness_summary import main as metrics_summary_main
from Damage_metric_summary import main as damage_loss_evaluation_main
from Compute_WAE import main as WAE_main
from Fitness_test_metrics import main as test_metrics_main

freqs = np.arange(len(FREQUENCY_MAPPING))  # frequency indices to sweep over (0-based)
times = np.zeros((len(TYPES) + 1, len(freqs)))



def _beta_suffix(beta):
    return f"_beta{beta}" if beta != BETA_CONSTANT else ""


def _run_one(freq, type, raw_features, beta=BETA_CONSTANT):
    np.random.seed(42)
    run_bayesian_optimization(freq=freq, type=type, raw_features=raw_features, beta=beta)


def _run_in_subprocess(freq, type, raw_features, beta=BETA_CONSTANT):
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_one, args=(freq, type, raw_features, beta))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"Subprocess for type={type!r} freq={freq} raw_features={raw_features} beta={beta} "
            f"failed with exit code {p.exitcode}"
        )


def run_by_area_beta_sweep(freq):
    """Sweep beta_constant for type='by_area', raw_features=False, at a fixed freq."""
    for beta in BETAS:
        t_begin = time.perf_counter()
        _run_in_subprocess(freq, 'by_area', False, beta)
        t_end = time.perf_counter()
        print(f"beta={beta} took {t_end - t_begin:.2f}s")

    folders = [f"Bayesian_GCN_by_area_freq{freq}{_beta_suffix(beta)}" for beta in BETAS]
    out_dir = PROJECT_ROOT / "graph_performance_results_by_area_beta_sweep"
    graph_performance_beta_sweep_main(recompute=True, folders=folders, out_dir=out_dir)


def main():
    np.random.seed(42)
    for type in TYPES:
           
        for freq in freqs:
            if type not in OPTIMIZED_TYPES:
                continue
            t_begin = time.perf_counter()
            _run_in_subprocess(freq, type, False)
            t_end = time.perf_counter()
            times[TYPES.index(type), freqs.index(freq)] = t_end - t_begin
            print(f"Time for {type} freq={freq}: {t_end - t_begin:.2f}s\n")

        if type == 'basic':
            folders = [f"Bayesian_GCN_freq{freq}" for freq in freqs]
        else:
            folders = [f"Bayesian_GCN_{type}_freq{freq}" for freq in freqs]


        if type == 'without_map_loss':
            out_dir = PROJECT_ROOT / f"graph_performance_results_wml"
        else:
            out_dir = PROJECT_ROOT / f"graph_performance_results_{type}"

        # Generate a performance summary for this type across all frequencies
        graph_performance_main(recompute=True, folders=folders, out_dir=out_dir)


    for freq in freqs:
        if not OPTIMIZE_RAW:
            continue  
        t_begin = time.perf_counter()
        _run_in_subprocess(freq, 'basic', True)
        t_end = time.perf_counter()
        times[len(TYPES), freqs.index(freq)] = t_end - t_begin
        print(f"Time for raw features freq={freq}: {t_end - t_begin:.2f}s\n")

    folders = [f"Bayesian_GCN_raw_freq{freq}" for freq in freqs]
    out_dir = PROJECT_ROOT / "graph_performance_results_raw"
    graph_performance_main(recompute=True, folders=folders, out_dir=out_dir, raw_features=True)

    WAE_main()              # compute weighted average metrics
    test_metrics_main()     # compute test metrics 
    metrics_summary_main()  # make Fitness summary plots
    damage_loss_evaluation_main() # compute damage loss evaluation metrics
    
    np.random.seed(42)
    for i, type in enumerate(TYPES + ['raw']):
        print(f"Times for {type}: {times[i, :]}")
        avg_time = np.mean(times[i, :])
        print(f"Average time for {type}: {avg_time:.2f} seconds")

    


if __name__ == "__main__":
    main()
    run_by_area_beta_sweep(FREQ_FOR_BETA_SWEEP)
