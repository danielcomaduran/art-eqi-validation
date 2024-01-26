"""
    Compilation of functions to work with the data files.
"""

from typing import Any
import json
import numpy as np
import pandas as pd
import pyxdf
import mne
import os
import random

def import_xdf(file:str) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """ Imports XDF data and returns `lsl_time`, `lsl_markers`, 
        `eeg_time`, and `eeg_data`. """

    streams,_ = pyxdf.load_xdf(file)
    
    # Detect EEG and LSL streams
    for stream in streams:
        stream_type = stream["info"]["type"][0]

        if stream_type == "LSL_Marker_Strings":
            lsl_time = stream["time_stamps"]
            lsl_markers = stream["time_series"]            
        elif stream_type == "EEG":
            eeg_time = stream["time_stamps"]
            eeg_data = stream["time_series"]

            # Make sure data output is always [channels, samples]
            eeg_data_shape = np.shape(eeg_data)
            if eeg_data_shape[0] > eeg_data_shape[1]:
                eeg_data = eeg_data.T

    return (lsl_time, lsl_markers, eeg_time, eeg_data)

def raw_to_epochs(
        eeg_time:list,
        eeg_markers:list,
        lsl_time:list,
        lsl_markers,
        times:list
        ) -> tuple[np.ndarray, np.ndarray]:
    """ 
    Creates EEG epochs of set `times [start, end]` in sec. 
    
    Returns
    -------
    - `eeg_epochs`: EEG epochs with shape [epoch, channels, samples]
    - `labels`: Labels of each epoch with length [epoch]. 
    """

def xdf_eeg_info(file:str):
    """ Returns `mne.info` object from .xdf `file`. """
    
    streams,_ = pyxdf.load_xdf(file)
    
    # Lookk for EEG stream to get info
    for stream in streams:
        stream_type = stream["info"]["type"][0]
      
        if stream_type == "EEG":
            # Channel names
            ch_names = []
            for ch_info in stream["info"]["desc"][0]["channels"][0]["channel"]:
                ch_name = ch_info["label"][0]
                ch_names.append(ch_name)
            
            # Sampling rate
            srate = float(stream["info"]["nominal_srate"][0])

            # Create info object
            mne_info = mne.create_info(
                ch_names = ch_names,
                sfreq = srate,
            )

            break

    return mne_info

def xdf_to_raw(
        file:str,
        exclude_chs:list = ["X1", "X2", "X3", "A1", "A2", "TRG"]):
    """ Returns MNE RAW object with EEG data from .xdf `file`. """
    
    # Import EEG times and data
    [_,_,eeg_times, eeg_data] = import_xdf(file)

    # Extract information from xdf
    info = xdf_eeg_info(file)

    # Create MNE raw object
    raw = mne.io.RawArray(
        data = eeg_data,
        info = info
    )

    # Remove channels in excluding list
    raw = raw.copy().drop_channels(
        ch_names = exclude_chs,
        on_missing = "ignore"
        )

    return raw

def xdf_to_mne():
    pass
            

def append_json(
        filename:str,
        subject:str,
        key:str,
        value:Any
        ):
    """ Appends or creates `key`:`value` to `subject` in `filename` """

    # Open the json file and load its data. If file does not exist, create it
    filename_abspath = os.path.abspath(filename)

    try:
        with open(filename_abspath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        with open(filename_abspath, "w") as f:
            data = {}
                
    # Check if the subject exists in the data
    if subject in data:
        # Append the key-value pair to the subject
        data[subject][key] = value
    else:
        # Create a new subject with the key-value pair
        data[subject] = {key: value}
    
    # Write the updated data to the json file
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    
def split_list(lst, test_percentage, seed:int|None=None):
    """" 
        Returns `[optimization, test]` indices based on the `test_percentage` variable.
        For example, if you have a list with 5 elements, and `test_percentage=20`, 
        will return 4 indices for `optimization` (i.e., 80\%), and  1 index for 
        test (i.e., 20\%)
        
        Note: use `seed` for repeatible results . 
    """
    
    test_size = int(len(lst) * test_percentage / 100)

    
    if (seed is not None):
        random.seed = seed
    
    random.shuffle(lst)
    optimization = lst[:-test_size]
    test = lst[-test_size:]
    
    return optimization, test