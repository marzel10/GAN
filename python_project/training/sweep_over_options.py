import time
import numpy as np
from BO_GCN import run_bayesian_optimization
from config import PROJECT_ROOT
from graph_performance import main as graph_performance_main

types = [ "by_area", "geometry", "peak"]
freqs = [ 1, 2, 3, 4, 5]
times = np.zeros((len(types)+1, len(freqs)))


for type in types:
    
    for freq in freqs:
        t_begin = time.perf_counter()
        run_bayesian_optimization(freq=freq, type=type, raw_features=False)
        t_end = time.perf_counter()
        times[types.index(type), freqs.index(freq)] = t_end - t_begin
    if type == 'basic':
        folders = [f"Bayesian_GCN_freq{freq}" for freq in freqs]
    else:
        folders = [f"Bayesian_GCN_{type}_freq{freq}" for freq in freqs]

    out_dir = PROJECT_ROOT / f"graph_performance_results_{type}"


    #graph_performance_main(recompute=True, folders=folders, out_dir=out_dir)


for freq in freqs:
    t_begin = time.perf_counter()
    run_bayesian_optimization(freq=freq, type='basic', raw_features=True)
    t_end = time.perf_counter()
    times[len(types), freqs.index(freq)] = t_end - t_begin

folders = [f"Bayesian_GCN_raw_freq{freq}" for freq in freqs] 
out_dir = PROJECT_ROOT / f"graph_performance_results_raw"
#graph_performance_main(recompute=True, folders=folders, out_dir=out_dir)

for i, type in enumerate(types + ['raw']):
    print(f"Times for {type}: {times[i, :]}")
    avg_time = np.mean(times[i, :])
    print(f"Average time for {type}: {avg_time:.2f} seconds")