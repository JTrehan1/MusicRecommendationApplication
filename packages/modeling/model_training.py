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