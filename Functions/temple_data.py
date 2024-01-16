
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
        self.montage_type = montage_type
    
        # Create EDF object with data
        
        self.edf = io.read_raw_edf(self.file_name)

    def set_montage(self):
        """ Creates an array mask to obtain the proper montage """

    def get_data(self):
        """ Returns full trial with proper montage set. """

    def get_clean_data(self):
        """ Returns a list of windows with all the windows that are not
            asociated to an artifact (i.e., clean windows). """

    def get_artifacts(self, artifact_file:str, artifact_type:str):
        """ Returns a list with all the artifacts of type `artifact_type` """

    def list_to_np(self, data:list):
        """ Trims `data` list to the shortest element and returns a numpy
            array where the first dimension = len(data). """

