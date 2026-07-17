# Model training script which takes the CSR matrix as input for an ALS or BPR model.

from dataclasses import dataclass
import logging
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hyperparameters that must always be supplied for any model type.
REQUIRED_HYPERPARAMETERS = {"factors", "regularization", "iterations"}

# Registry of supported model types: which class to train, which config.yaml section
# holds its default hyperparameters, and which hyperparameter keys it accepts.
# accepted_keys mirrors the constructor parameters documented in implicit's
# AlternatingLeastSquares/BayesianPersonalizedRanking docstrings (their __init__ signatures
# aren't introspectable since they're Cython-generated (*args, **kwargs) wrappers).
MODEL_REGISTRY = {
    "ALS": {
        "model_class": AlternatingLeastSquares,
        "config_key": "model_training_als",
        "accepted_keys": {
            "factors", "regularization", "alpha", "dtype", "use_native", "use_cg",
            "use_gpu", "iterations", "calculate_training_loss", "num_threads", "random_state",
        },
    },
    "BPR": {
        "model_class": BayesianPersonalizedRanking,
        "config_key": "model_training_bpr",
        "accepted_keys": {
            "factors", "learning_rate", "regularization", "dtype", "use_gpu",
            "iterations", "verify_negative_samples", "num_threads", "random_state",
        },
    },
}


@dataclass
class ModelTrainingOutput:
    trained_model: Any
    model_type: str
    track_uri_to_index: dict
    index_to_track_uri: dict
    hyperparameters: dict


class ModelTraining:
    """Class for training ALS and BPR collaborative filtering models. Hyperparameters are sourced
    from config.yaml by default, or may be passed explicitly (used by HyperparameterTuning to try
    trial-specific hyperparameters). Run using dedicated orchestration script.
    """

    def __init__(self, csr_matrix, track_uri_to_index: dict, index_to_track_uri: dict, config: dict):
        """Stores the training CSR matrix, track URI mappings, and configuration as instance variables.

        Args:
            csr_matrix: The user-item (playlist-track) interaction matrix to train on.
            track_uri_to_index (dict): Mapping of track_uri to matrix column index.
            index_to_track_uri (dict): Mapping of matrix column index to track_uri.
            config (dict): The full application config (as loaded from config.yaml), used to look up
                default hyperparameters for each model type."""
        self.csr_matrix = csr_matrix
        self.track_uri_to_index = track_uri_to_index
        self.index_to_track_uri = index_to_track_uri
        self.config = config

    def get_hyperparameters(self, model_type: str) -> dict:
        """Look up and validate the default hyperparameters for model_type from config.yaml.

        Args:
            model_type (str): The type of model to look up, either 'ALS' or 'BPR'.
        Returns:
            dict: The validated hyperparameters for the model type."""
        self._validate_model_type(model_type)
        config_key = MODEL_REGISTRY[model_type]["config_key"]
        if config_key not in self.config:
            raise ValueError(f"Missing '{config_key}' section in configuration for model type '{model_type}'.")

        hyperparameters = dict(self.config[config_key])
        self._validate_hyperparameters(model_type, hyperparameters)
        return hyperparameters

    def _validate_model_type(self, model_type: str) -> None:
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Invalid model type specified: {model_type}. Must be one of {sorted(MODEL_REGISTRY)}.")

    def _validate_hyperparameters(self, model_type: str, hyperparameters: dict) -> None:
        accepted_keys = MODEL_REGISTRY[model_type]["accepted_keys"]
        missing_keys = REQUIRED_HYPERPARAMETERS - hyperparameters.keys()
        unexpected_keys = hyperparameters.keys() - accepted_keys
        if missing_keys:
            raise ValueError(f"Missing required hyperparameters {sorted(missing_keys)} for model type '{model_type}'.")
        if unexpected_keys:
            raise ValueError(
                f"Unexpected hyperparameters {sorted(unexpected_keys)} for model type '{model_type}'. "
                f"Accepted keys: {sorted(accepted_keys)}."
            )

    def train(self, model_type: str, hyperparameters: Optional[dict] = None) -> ModelTrainingOutput:
        """Trains the model based on the specified model type and returns the trained model along with
        relevant metadata. Evaluation is handled separately by ModelEvaluation.

        Args:
            model_type (str): The type of model to train, either 'ALS' or 'BPR'.
            hyperparameters (dict, optional): Hyperparameters to train with. Defaults to config.yaml's
                values for model_type when not supplied.
        Returns:
            ModelTrainingOutput: A dataclass containing the trained model, model type, track URI
                mappings, and the hyperparameters used."""
        self._validate_model_type(model_type)

        if hyperparameters is None:
            hyperparameters = self.get_hyperparameters(model_type)
        else:
            self._validate_hyperparameters(model_type, hyperparameters)

        logger.info(f"Starting model training for model type: {model_type} with hyperparameters: {hyperparameters}")

        model_class = MODEL_REGISTRY[model_type]["model_class"]
        model = model_class(**hyperparameters)
        model.fit(self.csr_matrix)

        logger.info(f"Model training completed for model type: {model_type}")

        return ModelTrainingOutput(
            trained_model=model,
            model_type=model_type,
            track_uri_to_index=self.track_uri_to_index,
            index_to_track_uri=self.index_to_track_uri,
            hyperparameters=hyperparameters,
        )
