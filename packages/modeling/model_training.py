# Model training script which takes the CSR matrix as input for an ALS model 

from dataclasses import dataclass
import logging
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import precision_at_k, mean_average_precision_at_k 

logger = logging.getLogger(__name__)

@dataclass
class ModelTrainingOutput:
    trained_model: dict 
    track_uri_to_index: dict
    index_to_track_uri: dict
    hyperparameters: dict
    evaluation_scores: dict 
    