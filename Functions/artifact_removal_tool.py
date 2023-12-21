"""
    Artifact Removal Tools
    -----------------
    These functions are used to remove artifacts from preprocessed EEG data

"""

## Import libraries
import warnings
import numpy as np
from scipy.stats import kurtosis
import scipy.linalg as linalg
from sklearn.cluster import KMeans
from numpy import matlib as matlib
import concurrent.futures as cf

class ART:
    """
        Artifact removal tool to remove eyeblink artifacts from EEG data using an SSA 
        approach implementation. Code is adapted from `[Maddirala & Veluvolu 2021]`
    """

    def __init__(
            self,
            window_length:int | float = 125,
            n_clusters:int = 4,
            fd_threshold:float = 1.4,
            ssa_threshold:float = 0.01,
            var_tol:float = 1e-15,
            antidiag_method:str = "mask",
            svd_method:str = "sci"
            ):
        """
            Initialize new instance of Artifact Removal Tool (ART).

            Attributes
            ----------
                - `window_length`: int | float, optional\n
                    Length of window used to create the SSA matrix:\n
                        - If "float": length must be in [msec] 
                        - If "int": length is the number of [sample points]
                - `n_clusters`: int, optional\n
                    Number of k-clusters used in KNN classification [A.U.]
                - `fd_threshold`: float, optional\n                 
                    Fractal dimension threshold [A.U.]
                - `ssa_threshold`: float, optional\n
                    Singular Spectrum Analysis threshold.\n
                    Eigenvalues > ssa_threshold are included in SSA [A.U.]
                - `var_tol`: float, optional\n
                    Threshold for variance tolerance. Values below this number will
                    be considered to be 0.
                - `antidiag_method`:str, optional\n
                    Method to use for antidiagonal matrix computation. Options are:\n
                        - `mask`: Compute matrix with antidiagonals and boolean mask,
                        calculate mean of matrix
                        - `simple`: Iterate through each antidiagonal and
                        calculate mean value  
                - `svd_method`: str, optional
                    Method to use for SVD calculation:
                        - `sci`: Scipy
                        - `np`: Numpy 
        """
        # Public attributes
        self.n_clusters = n_clusters
        self.fd_threshold = fd_threshold
        self.ssa_threshold = ssa_threshold
        self.var_tol = var_tol
        self.antidiag_method = antidiag_method
        self.svd_method = svd_method
        
        # General flags
        self.artifact_found = False
        self.null_var_found = False
        
        # Private attributes
        self._window_length = window_length
        
    def remove_artifacts(self, eeg_data, srate) -> np.ndarray:
        """ Removes artifact from `eeg_data` and returns clean time series. """

        self.eeg_data = eeg_data
        self.srate = srate

        # Remove artifacts from each channel
        nchans = self._eeg_data.shape[0]
        self.eeg_clean = np.zeros_like(self._eeg_data)
        for channel in range(nchans):
            self.eeg_clean[channel,:] = self.single_channel_artifact_removal(
                self._eeg_data[channel,:],
            )

        return self.eeg_clean

    def single_channel_artifact_removal(self, eeg_data) -> np.ndarray:
        """ Removes artifacts from a single channel. """      
        
        # Compute embedded matrix 
        eeg_embedded = self.embedded_matrix(eeg_data)

        # If null variance is found, return raw data
        self.null_variance(eeg_embedded)
        if self.null_var_found:
            return eeg_data

        # Calculate features from embedded matrix
        eeg_features = self.compute_features(eeg_embedded)

        # Kmeans decomposition
        eeg_components = self.compute_kmeans(
            eeg_embedded,
            eeg_features
        )

        # Fractal dimension 
        fd_mask = self.compute_fd_mask(eeg_components)

        # If no artifacts were found return original data
        if not self.artifact_found:
            return eeg_data
        
        # Create time series with artifact mask
        eeg_mask = np.sum(eeg_components[fd_mask,:],0)  # Vector with artifact points != 0
        # -  Multiply mask with original to get eeg_artifact with correct values [V]
        eeg_sub_artifact = eeg_data*(eeg_mask != 0).astype(int)

        # Compute singular spectrum analysis (SSA)
        eeg_artifact = self.compute_ssa(eeg_sub_artifact)

        # Subtract artifact from signal to get clean EEG
        eeg_clean = eeg_data - eeg_artifact

        return eeg_clean

    def embedded_matrix(self, eeg_data) -> np.ndarray:
        """ Sets EEG embedded matrix from single EEG time series. """
    
        # - n = length of time series
        self.n = eeg_data.shape[-1]

        # - m = Number of rows
        if type(self.window_length) == int:
            self.m = self.window_length
        elif type(self.window_length) == float:
            self.m = np.floor(self.window_length * self.srate).astype(int)
            
        # - k = Number of columns
        self.k = self.n - self.m + 1

        # - Create indexing matrix for embedding EEG data
        idx_col = np.arange(self.k).reshape(1,-1).repeat(self.m, axis=0)
        idx_row = np.arange(self.m).reshape(-1,1).repeat(self.k, axis=1)
        self._idx_mat = idx_col + idx_row

        eeg_embedded = eeg_data[self._idx_mat]

        return eeg_embedded

    def null_variance(self, eeg_embedded):
        """ 
            Evaluates if there are windows with null variance (i.e., flat lines). 
            This can happen when the signal clips or is constant around 0.

            If this happens, the kurosis will not be able to be calculated.

            Returns `true` if variance < var_tol is found for any column of the 
            index matrix.                
        """

        if np.any(eeg_embedded.var(axis=0) < self.var_tol):
            warnings.warn("Null variance found, returning raw EEG")
            self.null_var_found = True            
    
    def compute_features(self, eeg_embedded) -> np.ndarray:
        """ 
            Sets a matrix with energy, Hjorth mobility, Kurtosis, and 
            amplitude range for each column of the embedded matrix.
        """

        # Compute features
        f1 = (eeg_embedded**2).sum(axis=0)              # Energy [V^2]
        f2 = np.sqrt((np.diff(eeg_embedded,axis=0)).var(axis=0)) \
             / eeg_embedded.var(axis=0)                 # H_mobility
        f3 = kurtosis(eeg_embedded, axis=0)             # Kurtosis
        f4 = eeg_embedded.max(0) - eeg_embedded.min(0)  # Range
        
        # Concatenate features
        eeg_features = np.array((f1,f2,f3,f4))

        return eeg_features

    def compute_kmeans(self, eeg_embedded, eeg_features) -> np.ndarray:
        """
            Computes k-means decomposition. Signal is decomposed into k-groups. 
            Clusters are created from each k-group and a time series is reconstructed
            using antidiagonal averaring.

            Returns
            -------
                - `eeg_components`: k-clusters components with same length as eeg_data.
        """
        
        # Perform Kmeans classification
        kmeans = KMeans(
            n_init="auto",
            n_clusters=self.n_clusters,
            )
        kmeans.fit(eeg_features.T)

        # Compute decomposed matrices
        # - Preallocate variable for n_clusters EEG components
        eeg_component = np.zeros((self.n_clusters, self.n))

        # - Calculate EEG component for each cluster
        for cluster in range(self.n_clusters):
            # - Temp matrix to store decomposed EEG for each cluster
            eeg_decomposed = np.zeros((self.m,self.k))    
            
            # - Determine columns to copy based on the kmeans label
            copy_index = (kmeans.labels_== cluster) 
            eeg_decomposed[:,copy_index] = eeg_embedded[:,copy_index]
                    
            # Reconstruct signal from antidiagonal averages
            eeg_component[cluster, :] = self.mean_antidiag(
                eeg_decomposed,
                self.antidiag_method,
                )
            
        return eeg_component
    
    def compute_fd_mask(self, eeg_components):
        """ Computes fractal dimension mask. """
             
        # - Normalize EEG to unit square
        x_norm = np.repeat(
            np.reshape(np.linspace(0, 1, self.n), [-1,1]),
            self.n_clusters, axis=1)
        
        # -- 3D Matrix to store x_norm and y_norm [sample x [x,y] x n_cluster]
        y_num = eeg_components - matlib.repmat(
            np.amin(eeg_components, axis=1, keepdims=True),
            1,
            self.n
            )
        y_den = matlib.repmat(np.amax(eeg_components, axis=1, keepdims=True) 
                              - np.amin(eeg_components, axis=1, keepdims=True), 1, self.n)
        y_norm = np.divide(y_num, y_den).T
        z = np.array([x_norm, y_norm]) 

        # - Calculate length of signal (l2-norm for each n_cluster) [A.U.]
        l = np.sum(np.sqrt(np.sum(np.square(np.diff(z, axis=1)), axis=0)), axis=0)  
        
        # - Fractal dimension value
        fd = 1 + (np.log(l) / np.log(2*(self.n-1)))

        # - Apply threshold to FD to determine artifact components
        fd_mask = fd < self.fd_threshold

        # - Set flag for artifacts found
        if fd_mask.any():
            self.artifact_found = True        

        return fd_mask
    
    def compute_ssa(self, eeg_sub_artifact) -> np.ndarray:
        """ 
            Computes singular spectrum analysis (SSA). 

            Parameters
            ----------
                - `eeg_sub_artifact`: MISSING DESCRIPTION

            Returns
            -------
                - `artifact`: EEG artifact with shape of RAW EEG
        """
        # Singular Value Decomposition
        # - Create multivariate matrix for each channel
        artifact_embedded = eeg_sub_artifact[self._idx_mat]       
        
        # - Use scipy or numpy for SVD calculation
        svd_methods = {
            "sci": linalg.svd(artifact_embedded, full_matrices=False),
            "np": np.linalg.svd(artifact_embedded, full_matrices=False)
        }

        [u, s, vh] = svd_methods[self.svd_method]

        # Keep only eigenvectors > ssa_threshold and reconstruct signal
        eigen_ratio = (s / s.sum()) > self.ssa_threshold 
        artifact_sub = u[:,eigen_ratio] @ np.diag(s[eigen_ratio]) @ vh[eigen_ratio,:]
        artifact = self.mean_antidiag(artifact_sub, self.antidiag_method)

        return artifact
        


    def mean_antidiag(self, input_mat, method):
        """
            This function returns the mean of the antidiagonal components of a matrix

            Parameters
            ----------
                input_mat: array_like
                    Matrix with shape [i,k] for which the mean of the antidiagonal 
                    components will be calculated. Must be a 2D matrix.
                method: str
                    Method used to calculate average of antidiagonals
                    'mask': Compute matrix with antidiagonals and boolean mask, calculate mean of matrix
                    'simple': Iterate through each antidiagonal and calculate mean value
                    
            Returns
            -------
                average_vect: array_like
                    1D vector containing the mean values of the antidiagonal components
        """

        input_shape = input_mat.shape       # Shape of input matrix
        input_flip = np.fliplr(input_mat)   # Flip input matrix from left to right

        # Calculate mean across diagonals
        if method == 'simple':
            average_vect = np.zeros(self.n)  # Preallocate vector with average values

            for i_diag, k_diag in enumerate(range(-input_shape[0]+1,input_shape[1])):
                average_vect[-i_diag-1] = input_flip.diagonal(offset=k_diag).mean() # Organize values from end to start to get right order

        elif method == 'mask':
            max_diag = (input_flip.diagonal(offset=0)).size # Size of longest diagonal
            diag_mat = np.zeros((max_diag,self.n))               # Empty matrix to store antidiagonal values
            mask_mat = np.zeros((max_diag,self.n))               # Empty matrix to store mask values

            for i, k_diag in enumerate(range(-input_shape[0]+1, input_shape[1])):
                diag_vals = input_flip.diagonal(offset=k_diag)  # Values of the k^th diagonal
                n_diag = diag_vals.size                         # Length of values of the k^th diagonal
                diag_mat[0:n_diag,i] = diag_vals
                mask_mat[0:n_diag,i] = 1

            average_vect = np.flip(diag_mat.mean(axis=0, where=mask_mat.astype(bool)))

        else:
            print('Antidiagonal method not available')
            return None

        return average_vect

    @property
    def eeg_data(self):
        """ Getter method for 'eeg' property. """
        return self._eeg_data

    @eeg_data.setter
    def eeg_data(self, value):
        """ Setter method for 'eeg' property. """
        if ((value.ndim == 1) or  (value.shape[-2] < value.shape[-1])):
            self._eeg_data = np.atleast_2d(value)
        else:
            raise ValueError(
                "EEG must be single dimension or in end in shape `[channels, samples]`"
                )
        
    @property
    def window_length(self):
        """ Getter method for 'window_length' property. """
        return self._window_length

    @window_length.setter
    def window_length(self, value):
        """ Setter method for 'window_length' property. """
        if isinstance(value, (int,float)):
            self._window_length = value
        else:
            raise ValueError("Window length must be type `int` or `float`")