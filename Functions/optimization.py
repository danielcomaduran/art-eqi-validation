## Import libraries
# - Default libraries
import copy
import numpy as np
import scipy.optimize as optimize
from rich import print as rprint
from rich.pretty import pprint as rpprint

# Custom libraries
from Functions.artifact_removal_tool import ART
import Functions.eeg_quality_index as eqi

def maximize_eqi(x, *args):
    # Separate input variables
    [n_clusters, fd_threshold, ssa_threshold] = copy.deepcopy(x)
    [clean_data, artifact_data, srate, window_length] = args

    n_clusters = int(n_clusters)

    # Convert window time [sec] to number of samples [n]
    window_samples = int(window_length * srate)

    # Creat artifact removal object
    art = ART(
        window_length = window_length,
        n_clusters = n_clusters,
        fd_threshold = fd_threshold,
        ssa_threshold = ssa_threshold 
    )

    # Apply artifact removal
    test_data = art.remove_artifacts(
        eeg_data = artifact_data,
        srate = srate
    )

    eqi_scoring = eqi.scoring(
        clean_eeg = clean_data,
        test_eeg = test_data,
        srate_clean = srate,
        srate_test = srate,
        window = int(window_samples // 10),
        slide = int(window_samples // 20)
    )

    # rpprint({
    #     "n_clusters": n_clusters,
    #     "fd_threshold": fd_threshold,
    #     "ssa_threshold": ssa_threshold,
    #     "eqi_scoring[0]": eqi_scoring[0]
    # })

    eqi_total = eqi_scoring[0]

    # Use the complement to minimize the problem
    eqi_total_complement = 100 - np.mean(eqi_total)
    # print(f" EQI value {eqi_total_complement}")

    return eqi_total_complement

# def optimize_to_eqi(x, *args):
#     [x0, fval] = optimize.brute(maximize_eqi, x, args)

#     return [x0, fval]

def optimize_to_eqi(x0, bounds, args):
    # print(x0, bounds, args)

    res = optimize.minimize(
        maximize_eqi,
        x0 = x0,
        bounds = bounds,
        args = args,
        method="L-BFGS-B",
        callback = lambda:print("Running")
    )
    