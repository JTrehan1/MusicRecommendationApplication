from dataclasses import dataclass
from packages.data_ingestion.data_ingestion import IngestionOutput

import scipy.sparse as sp
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataSplittingOutput:
    """Dataclass to encapsulate the output of the data splitting process, including the training and testing datasets."""
    train_data: sp.csr_matrix
    test_data: sp.csr_matrix
    
class DataSplitting:
    
    def __init__(self, ingestion_output: IngestionOutput):
        self.sparse_matrix = ingestion_output.sparse_matrix
        
    def load_splitting_config(self, config: dict):
        """Load the configuration for data splitting from a dictionary."""
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)
        