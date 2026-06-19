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
    
    def __init__(self, csr_matrix, track_uri_to_index, index_to_track_uri):
        """Stores the CSR matrix as an instance variable to be used for model training."""
        self.csr_matrix = csr_matrix
        self.track_uri_to_index = track_uri_to_index
        self.index_to_track_uri = index_to_track_uri
    
        
    def train(self, model_type: str, hyperparameters: dict) -> ModelTrainingOutput:
        """Trains the model based on the specified model type and returns the trained model along with relevant metadata and evaluation scores. 
        Args:
            model_type (str): The type of model to train, either 'ALS' or 'BPR'.
        Returns:
            ModelTrainingOutput: A dataclass containing the trained model, model type, track URI mappings, hyperparameters, and evaluation scores."""
        
        if model_type == "ALS":
            # Train ALS model
            model = AlternatingLeastSquares(**hyperparameters)
            model.fit(self.csr_matrix)
        elif model_type == "BPR":
            model = BayesianPersonalizedRanking(**hyperparameters)
            model.fit(self.csr_matrix)
        else:
            raise ValueError(f"Invalid model type specified: {model_type}. Must be either 'ALS' or 'BPR'.")
        
        logger.info(f"Model training completed for model type: {model_type} with hyperparameters: {hyperparameters}")   
        
        return ModelTrainingOutput(
            trained_model=model,
            model_type=model_type,
            track_uri_to_index=self.track_uri_to_index,
            index_to_track_uri=self.index_to_track_uri,
            hyperparameters=hyperparameters,
            evaluation_scores={} # Placeholder for evaluation scores, to be populated in future iterations
        )