## Import libraries
# - Default libraries
import numpy as np
import scipy.optimize as optimize

# Custom libraries
from Functions.artifact_removal_tool import ART
import Functions.eeg_quality_index as eqi

def maximize_eqi(x, *args):
    # Separate input variables
    [n_clusters, fd_threshold, ssa_threshold] = x
    [clean_data, artifact_data, srate, window_length] = args

    # Convert window time [sec] to number of samples [n]
    window_samples = int(window_length * srate)

    art = ART(
        window_length = window_length,
        n_clusters = n_clusters,
        fd_threshold = fd_threshold,
        ssa_threshold = ssa_threshold 
    )

    eqi_total = eqi.scoring(
        clean_eeg = clean_data,
        test_eeg = artifact_data,
        srate_clean = srate,
        srate_test = srate,
        window = window_samples,
        slide = window_samples // 2
    )[0]

    # Use the complement to minimize the problem
    eqi_total_complement = 100 - eqi_total

    return eqi_total_complement

def optimize_to_eqi(x, *args):
    [x0, fval] = optimize.brute(maximize_eqi, x, args)
    