# Model training script which takes the CSR matrix as input for an ALS model 

from dataclasses import dataclass
import logging
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import precision_at_k, mean_average_precision_at_k 
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class ModelTrainingOutput:
    trained_model: Any
    model_type: str 
    track_uri_to_index: dict
    index_to_track_uri: dict
    hyperparameters: dict
    evaluation_scores: dict 

class ModelTraining:
    """Class for containing the model training of both ALS and BPR models. Run using dedicated orchestration script. 
    """
    
    def __init__(self, csr_matrix):
        """Stores the CSR matrix as an instance variable to be used for model training."""
        self.csr_matrix = csr_matrix
        
    def train(model_type: str):
        """Trains the model based on the specified model type and returns the trained model along with relevant metadata and evaluation scores. 
        Args:
            model_type (str): The type of model to train, either 'ALS' or 'BPR'.
        Returns:
            ModelTrainingOutput: A dataclass containing the trained model, model type, track URI mappings, hyperparameters, and evaluation scores."""
        
        if model_type == "ALS":
            # Train ALS model
            pass 
        elif model_type == "BPR":
            # Train BPR model
            pass 
        else:
            raise ValueError(f"Invalid model type specified: {model_type}. Must be either 'ALS' or 'BPR'.")