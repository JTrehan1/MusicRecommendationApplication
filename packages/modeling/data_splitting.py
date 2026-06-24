from dataclasses import dataclass
from packages.data_ingestion.data_ingestion import IngestionOutput

import scipy.sparse as sp
import logging
import numpy as np


logger = logging.getLogger(__name__)

@dataclass
class DataSplittingOutput:
    """Dataclass to encapsulate the output of the data splitting process, including the training and testing datasets."""
    train_data: sp.csr_matrix
    test_data: sp.csr_matrix
    
class DataSplitting:
    
    def __init__(self, ingestion_output: IngestionOutput):
        self.sparse_matrix = ingestion_output.sparse_matrix
        
    # def load_splitting_config(self, config: dict):
    #     """Load the configuration for data splitting from a dictionary."""
    #     self.test_size = config.get("test_size", 0.2)
    #     self.random_state = config.get("random_state", 42)
    
    def csr_splitter(self, test_size: float = 0.2, random_state: int = 42) -> DataSplittingOutput:
        """Split the CSR matrix into training and testing datasets based on the specified test size and random state.
        
        Args:
            test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
            random_state (int): The seed used by the random number generator for reproducibility (default is 42)."""     
            
        train_matrix = []
        test_matrix =  [] 
        
        rng = np.random.default_rng(seed=random_state)
        
        # For each playlist row, find all the columns with non-zero values (i.e., the tracks that are present in the playlist).
        for row in self.sparse_matrix:
            number_tracks = (row !=0).sum() 
            if number_tracks < 5:
                logger.warning(f"Playlist with less than 5 tracks found. Playlist will be skipped in the split. Number of tracks: {number_tracks}")
                # Add the row to the training matrix, it will not be split 
                train_matrix.append(row)
                # Add a row of zeros to the test matrix to maintain the same shape as the training matrix 
                test_matrix.append(sp.csr_matrix(row.shape))            
            else:
                non_zero_indices = row.nonzero()[1]  # Get the column indices of non-zero elements
                # Randomly select non-zero column indices from the row to be includes in the test set based on test size and random state 
                n_holdout = int(np.floor(test_size * len(non_zero_indices)))  
                test_row_indices = rng.choice(non_zero_indices, size=n_holdout, replace=False)  
                test_matrix.append(sp.csr_matrix((np.ones(n_holdout), (np.zeros(n_holdout), test_row_indices)), shape=row.shape))
                train_row = row.copy()
                train_row[0, test_row_indices] = 0  # Set the selected test indices to zero in the training row
                train_row.eliminate_zeros()  # Remove any explicit zeros to maintain the sparse structure
                train_matrix.append(train_row)
                
                # Stack the rows to form the final training and testing matrices
        train_matrix = sp.vstack(train_matrix)
        test_matrix = sp.vstack(test_matrix)
                
        return DataSplittingOutput(train_data=train_matrix, test_data=test_matrix)
