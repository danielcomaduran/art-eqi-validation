## Import libraries
# - Default libraries
import tqdm
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
    [clean_data, artifact_data, srate, window_length, var_tol] = args

    n_clusters = int(n_clusters)

    # Convert window time [sec] to number of samples [n]
    window_samples = int(window_length * srate)

    # Creat artifact removal object
    art = ART(
        window_length = int(srate // 2),  # To have 500 msec windows like the paper
        n_clusters = n_clusters,
        fd_threshold = fd_threshold,
        ssa_threshold = ssa_threshold,
        var_tol = var_tol
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
        window = int(srate // 10),
        slide = int(srate // 20)
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
    
def simulated_annealing_optimizer(
    func,
    bounds,
    args,
    callback = lambda intermediate_result: None,
    display = True,
    max_iter = 100,
    max_stag_count = 20,
    x_err_tol = 0.1,
    MR = 0.5
    ):

    # Configurables
    max_stag_count = max_stag_count # maximum number of iterations where x_err hasn't changed
    max_iter = max_iter # maximum number of iterations
    x_err_tol = x_err_tol # tolerance for x_err (stop searching for a better solution when x_err < x_err_tol)
    MR = MR # mutation rate

    # Simulated annealing params
    T = 1
    alpha = 0.99
    beta = 1
    update_iters = 1

    def generate_candidate(bounds=bounds, args = args):
        x = [(np.random.uniform(bounds[d][0], bounds[d][1])) for d in range(len(bounds))]
        return x, func(x, *args)

    def generate_neighboring_candidate(x, MR=MR, bounds=bounds, args = args):
        x_new = copy.deepcopy(x)
        for i in range(len(x)):
            if np.random.uniform(0, 1) < MR:
                x_new[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return x_new, func(x_new, *args)

    x, x_err = generate_candidate(bounds, args) # generate a random x (particle/candidate) and its error (cost/objective/fitness function value)
    x_best = x.copy() # best solution found so far
    x_best_err = x_err # best error for that solution

    prev_x_err = x_err # previous iteration's x_err
    stag_count = 0 # number of iterations where x_err hasn't changed
    max_stag_count = max_stag_count # generate a new starting candidate if x_err hasn't changed for max_stag_count iterations

    # Stopping conditions
    max_iter = max_iter # maximum number of iterations
    x_err_tol = x_err_tol # stop searching for a better solution when x_err < x_err_tol

    pbar = tqdm() if display else None # progress bar
    n_iter = 1
    try:
        while (abs(x_best_err) > x_err_tol) and (n_iter < max_iter):
            
            if display:
                pbar.set_description(f'x_best_err: {x_best_err:.4f}')
                pbar.update(1)

            prev_x_err = x_err
            stag_count = stag_count + 1 if prev_x_err == x_err else 0

            x_new, x_new_err = generate_neighboring_candidate(
                x = x,
                MR = MR,
                bounds = bounds,
                args = args
                )

            if x_new_err <= x_err:
                x, x_err = x_new, x_new_err
                if x_err <= x_best_err:
                    x_best, x_best_err = x, x_err
            else:
                if np.random.uniform(0, 1) < np.exp(-beta * (x_new_err - x_err)/T):
                    x, x_err = x_new, x_new_err
                    
            if n_iter % update_iters == 0:
                T = alpha*T
                T = T if T > 0.01 else 0.01

            n_iter += 1

            callback((x_best, x_best_err))

    except KeyboardInterrupt:
        pass
    finally:
        pbar.close() if display else None

    print(f"x_best_err: {x_best_err:.4f} after {n_iter} iterations")

    return x_best, x_best_err

def heuristic_optimizer(
        func,
        bounds,
        args,
        callback = lambda intermediate_result: None,
        display = True,
        max_iter = 100,
        max_stag_count = 20,
        x_err_tol = 0.1,
        MR = 0.5
        ):
    

    # Configurables
    max_stag_count = max_stag_count # maximum number of iterations where x_err hasn't changed
    max_iter = max_iter # maximum number of iterations
    x_err_tol = x_err_tol # tolerance for x_err (stop searching for a better solution when x_err < x_err_tol)
    MR = MR # mutation rate

    def generate_candidate(bounds=bounds, args = args):
        x = [(np.random.uniform(bounds[d][0], bounds[d][1])) for d in range(len(bounds))]
        return x, func(x, *args)

    def generate_neighboring_candidate(x, MR=MR, bounds=bounds, args = args):
        x_new = copy.deepcopy(x)
        for i in range(len(x)):
            if np.random.uniform(0, 1) < MR:
                x_new[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return x_new, func(x_new, *args)

    x, x_err = generate_candidate(bounds, args) # generate a random x (particle/candidate) and its error (cost/objective/fitness function value)
    x_best = x.copy() # best solution found so far
    x_best_err = x_err # best error for that solution

    prev_x_err = x_err # previous iteration's x_err
    stag_count = 0 # number of iterations where x_err hasn't changed
    max_stag_count = max_stag_count # generate a new starting candidate if x_err hasn't changed for max_stag_count iterations

    # Stopping conditions
    max_iter = max_iter # maximum number of iterations
    x_err_tol = x_err_tol # stop searching for a better solution when x_err < x_err_tol

    pbar = tqdm() if display else None # progress bar
    n_iter = 1
    try:
        while abs(x_best_err) > x_err_tol and n_iter < max_iter:
            
            if display:
                pbar.set_description(f'x_best_err: {x_best_err:.4f}')
                pbar.update(1)

            prev_x_err = x_err
            stag_count = stag_count + 1 if prev_x_err == x_err else 0
            
            # Generate a random neighbor of x and calculate its error
            if stag_count > max_stag_count:
                x_new, x_new_err = generate_candidate(bounds, args)
                stag_count = 0
            else:
                x_new, x_new_err = copy.deepcopy(x), x_err
                x_new, x_new_err = generate_neighboring_candidate(x = x_new, MR = MR, bounds=bounds, args=args)

            # Update x and x_err if the new solution is better
            if x_new_err < x_err:
                x, x_err = x_new, x_new_err # if x_new is better, update x and x_err
                if x_err < x_best_err:
                    x_best, x_best_err = x, x_err # if x_new is the best, update x_best and x_best_err

            n_iter += 1

            callback((x_best, x_best_err))

    except KeyboardInterrupt:
        pass
    finally:
        pbar.close() if display else None

    print(f"x_best_err: {x_best_err:.4f} after {n_iter} iterations")

    return x_best, x_best_err