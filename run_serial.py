'''
This is a simple serial execution script for running the SHM pipeline.
Not recommended to run at once, but shows the order of execution of the different steps in the pipeline.
'''

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from BO_AE import main as BO_AE_main
from path_performance import main as path_performance_main
from extract_shi import pre_compute_AE_output
from features_extractor import pre_compute_features_for_all_panels
from sensitivity_study import main as sensitivity_study_main, run_by_area_beta_sweep

BO_AE_main() # Run the Bayesian optimization for the autoencoders 
path_performance_main() # Run the path performance analysis for the autoencoders
pre_compute_AE_output() # Pre-compute the sHI values for all paths and states using the trained autoencoders
pre_compute_features_for_all_panels() # Pre-compute raw features for raw graph dataset (from sensitivity study)
sensitivity_study_main() # Runs BO on different graph types (including basic workflow), it also computes Fitness and Damage metric
run_by_area_beta_sweep() # Runs the beta sweep for the area-based weighting function (also for the sensitivity study)