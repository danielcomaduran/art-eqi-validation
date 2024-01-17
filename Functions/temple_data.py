# Import libraries
import csv
import mne.io as io
import numpy as np


class TempleData:
    """ Object to handle the Temple Artifact Dataset. """
    def __init__(
        self,
        file_name:str,
        montage_type:str,
        ):
    
        self.file_name = file_name
        self.trial_id = file_name.split("\\")[-1].split(".")[0]
        self.montage_type = montage_type
    
        # Create EDF object with data
        edf = io.read_raw_edf(self.file_name)
        self.srate = edf.info["sfreq"]
        self.data = edf.get_data()
        self.ch_names = edf.info["ch_names"]

    def set_montage(self):
        """ Creates an array mask to obtain the proper montage """
        montage_info = {
            "ar": {
                "FP1-F7": "EEG FP1-REF - EEG F7-REF",
                "F7-T3": "EEG F7-REF - EEG T3-REF",
                "T3-T5": "EEG T3-REF - EEG T5-REF",
                "T5-O1": "EEG T5-REF - EEG O1-REF",
                "FP2-F8": "EEG FP2-REF - EEG F8-REF",
                "F8-T4": "EEG F8-REF - EEG T4-REF",
                "T4-T6": "EEG T4-REF - EEG T6-REF",
                "T6-O2": "EEG T6-REF - EEG O2-REF",
                "A1-T3": "EEG A1-REF - EEG T3-REF",
                "T3-C3": "EEG T3-REF - EEG C3-REF",
                "C3-CZ": "EEG C3-REF - EEG CZ-REF",
                "CZ-C4": "EEG CZ-REF - EEG C4-REF",
                "C4-T4": "EEG C4-REF - EEG T4-REF",
                "T4-A2": "EEG T4-REF - EEG A2-REF",
                "FP1-F3": "EEG FP1-REF - EEG F3-REF",
                "F3-C3": "EEG F3-REF - EEG C3-REF",
                "C3-P3": "EEG C3-REF - EEG P3-REF",
                "P3-O1": "EEG P3-REF - EEG O1-REF",
                "FP2-F4": "EEG FP2-REF - EEG F4-REF",
                "F4-C4": "EEG F4-REF - EEG C4-REF",
                "C4-P4": "EEG C4-REF - EEG P4-REF",
                "P4-O2": "EEG P4-REF - EEG O2-REF"
            },

            "le": {
                "FP1-F7": "EEG FP1-LE - EEG F7-LE",
                "F7-T3":  "EEG F7-LE - EEG T3-LE",
                "T3-T5":  "EEG T3-LE - EEG T5-LE",
                "T5-O1":  "EEG T5-LE - EEG O1-LE",
                "FP2-F8": "EEG FP2-LE - EEG F8-LE",
                "F8-T4":  "EEG F8-LE - EEG T4-LE",
                "T4-T6":  "EEG T4-LE - EEG T6-LE",
                "T6-O2":  "EEG T6-LE - EEG O2-LE",
                "A1-T3":  "EEG A1-LE - EEG T3-LE",
                "T3-C3":  "EEG T3-LE - EEG C3-LE",
                "C3-CZ":  "EEG C3-LE - EEG CZ-LE",
                "CZ-C4":  "EEG CZ-LE - EEG C4-LE",
                "C4-T4":  "EEG C4-LE - EEG T4-LE",
                "T4-A2":  "EEG T4-LE - EEG A2-LE",
                "FP1-F3": "EEG FP1-LE - EEG F3-LE",
                "F3-C3":  "EEG F3-LE - EEG C3-LE",
                "C3-P3":  "EEG C3-LE - EEG P3-LE",
                "P3-O1":  "EEG P3-LE - EEG O1-LE",
                "FP2-F4": "EEG FP2-LE - EEG F4-LE",
                "F4-C4":  "EEG F4-LE - EEG C4-LE",
                "C4-P4":  "EEG C4-LE - EEG P4-LE",
                "P4-O2":  "EEG P4-LE - EEG O2-LE"
            },

            "ar_a": {
                "FP1-F7": "EEG FP1-REF - EEG F7-REF",
                "F7-T3": "EEG F7-REF - EEG T3-REF",
                "T3-T5": "EEG T3-REF - EEG T5-REF",
                "T5-O1": "EEG T5-REF - EEG O1-REF",
                "FP2-F8": "EEG FP2-REF - EEG F8-REF",
                "F8-T4": "EEG F8-REF - EEG T4-REF",
                "T4-T6": "EEG T4-REF - EEG T6-REF",
                "T6-O2": "EEG T6-REF - EEG O2-REF",
                "T3-C3": "EEG T3-REF - EEG C3-REF",
                "C3-CZ": "EEG C3-REF - EEG CZ-REF",
                "CZ-C4": "EEG CZ-REF - EEG C4-REF",
                "C4-T4": "EEG C4-REF - EEG T4-REF",
                "FP1-F3": "EEG FP1-REF - EEG F3-REF",
                "F3-C3": "EEG F3-REF - EEG C3-REF",
                "C3-P3": "EEG C3-REF - EEG P3-REF",
                "P3-O1": "EEG P3-REF - EEG O1-REF",
                "FP2-F4": "EEG FP2-REF - EEG F4-REF",
                "F4-C4": "EEG F4-REF - EEG C4-REF",
                "C4-P4": "EEG C4-REF - EEG P4-REF",
                "P4-O2": "EEG P4-REF - EEG O2-REF",
            }
        }

        # Temp variables
        original_data_shape = self.data.shape
        montage = montage_info[self.montage_type]

        # Preallocate empty array to store the transformed eeg data
        transformed_data = np.zeros([len(montage), original_data_shape[-1]])

        # Loop through the montage channel names
        for [c, [_, value]] in enumerate(montage.items()):
            minuend = value.split(" - ")[0]
            subtrahend = value.split(" - ")[1]
            
            iminuend = [i for [i, channel] in enumerate(self.ch_names) if channel == minuend][0]
            isubtrahend = [i for [i, channel] in enumerate(self.ch_names) if channel == subtrahend][0]
            
            transformed_data[c,:] = self.data[isubtrahend,:] - self.data[iminuend,:]
            
        # Assign values with new montage
        self.data = transformed_data
        self.ch_names = list(montage.keys())

    def get_clean_data(self):
        """ Returns a list of windows with all the windows that are not
            asociated to an artifact (i.e., clean windows). """

    def get_artifacts(self, artifact_file:str, artifact_type:str):
        """ Returns a list with all the artifacts of type `artifact_type` """

    def list_to_np(self, data:list):
        """ Trims `data` list to the shortest element and returns a numpy
            array where the first dimension = len(data). """
        
    def _artifact_lines(self, artifact_file:str):
        """ Returns the lines in `artifact_file` that are associated with 
            the proper `file_name`. """
        
        # Preallocate list for output
        artifact_lines = []
    
        # Look for matching lines in `artifact_file`
        with open(artifact_file, "r") as f:
            
            # Loop through each line in the file
            for line in f:
                line = line.strip()
                
                # Check that line is not empty and matches trial_id name
                if line and (line.split(",")[0] == self.trial_id):
                    artifact_lines.append(line)
        
        return artifact_lines

  
    


