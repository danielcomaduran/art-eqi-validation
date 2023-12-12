"""
    Compilation of functions to process the EEG data
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

def bandpass_filter(x:np.ndarray, n:int, fc:list, srate:float) -> np.ndarray
    """ Implements a zero-delay `n`-order digital Butterworth filter.
        Output has same shape as `x`. """
    sos = butter(
        N = n,
        Wn = fc,
        btype = 'bandpass',
        output = "sos",
        fs = srate
    )

    filtered_x = sosfiltfilt(sos, x)

    return filtered_x


