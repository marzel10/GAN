import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.preprocessing import Normalizer


def monotonicity_criterion(DI, test_mode=False, test_idx=None):
    # Check if DI is monotonically increasing across states for each panel

    if test_mode and test_idx is None:
        raise ValueError("test_idx must be provided when test_mode is True")

    
    M = len(DI)
    monotonicity=0
    
    for i in range(M):
        panel_monotonicity  = 0

        if test_mode and i != test_idx:
            continue  # Skip this panel if we're in test mode and it's not the test panel
        panel_DI = DI[i]
        N = len(panel_DI)
        
        for j in range(N):
            partial_panel_monotonicity  = 0
            sum_weights = 0

            for k in range(j+1, N):
                point_mono = np.sign(panel_DI[j] - panel_DI[k])*(k-j)
                partial_panel_monotonicity += point_mono
                sum_weights += (k-j)
            
            panel_monotonicity += partial_panel_monotonicity / sum_weights if sum_weights != 0 else 0

        panel_monotonicity = np.abs(panel_monotonicity/(N-1))
        monotonicity += panel_monotonicity

    if test_mode:
        return monotonicity / 1  # Only one panel is considered in test mode
    return monotonicity / M
           



def scale_exact(HI_list, minimum):
    """
    Downsample a single DI/HI curve to `minimum` length via interpolation.
    Only compresses if the curve is longer than `minimum`; shorter curves are
    left as-is. Mirrors Diversity-DeepSAD-vs-DTC-VAE's Prognostic_criteria.scale_exact.
    """
    HI_list = np.asarray(HI_list)
    if HI_list.size > minimum:
        arr_interp = interp1d(np.arange(HI_list.size), HI_list)
        arr_compress = arr_interp(np.linspace(0, HI_list.size - 1, minimum))
    else:
        arr_compress = HI_list
    return np.array(arr_compress)


def trendability_criterion(DI):

    M = len(DI)

    # Downsample every panel's DI curve to the length of the shortest one
    minimum = min(len(v) for v in DI)
    X = np.stack([scale_exact(v, minimum) for v in DI], axis=0)

    # Standardize the data to have zero mean and unit variance
    X = Normalizer().fit_transform(X)

    trendability_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            trendability_matrix[i, j] = abs(pearsonr(X[i], X[j])[0])

    return np.min(trendability_matrix)

def prognosability_criterion(DI, test_mode=False, test_idx=None):

    if test_mode and test_idx is None:
        raise ValueError("test_idx must be provided when test_mode is True")

    if isinstance(test_idx, (list, tuple)):
        raise ValueError("test_idx must be a single number, multiple test samples are not implemented")

    M = len(DI)
    prognosability = 0

    end_point_std = 0
    mean_HI_range = 0

    mean_end_value = 0

    for j in range(M): 
        if test_mode and j == test_idx:
            continue  # Skip the test panel   
        mean_end_value += DI[j][-1]
    mean_end_value /= M

    
    for i in range(M):
        end_point_std += np.abs(DI[i][-1] - mean_end_value)**2
        mean_HI_range += np.abs(DI[i][-1] - DI[i][0])

    mean_HI_range /= M

    if test_mode: 
        end_point_std = np.abs(DI[test_idx][-1] - mean_end_value)
    else:   
        end_point_std = np.sqrt(end_point_std / M)

    prognosability = np.exp(-end_point_std / mean_HI_range)

    return prognosability

