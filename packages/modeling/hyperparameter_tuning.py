# Hyperparameter tuning script which runs an Optuna search over the space defined in config.yaml,
# training and evaluating a candidate model on every trial.

from dataclasses import dataclass
import logging
from typing import Any

import optuna
import scipy.sparse as sp

from packages.modeling.model_evaluation import METRIC_REGISTRY, ModelEvaluation
from packages.modeling.model_training import ModelTraining, ModelTrainingOutput

logger = logging.getLogger(__name__)

# Optuna's own trial logging is noisy at INFO level for every one of potentially many trials.
optuna.logging.set_verbosity(optuna.logging.WARNING)

SUPPORTED_SEARCH_TYPES = {"int", "float", "categorical"}


@dataclass
class HyperparameterTuningOutput:
    model_type: str
    best_hyperparameters: dict
    best_score: float
    best_training_output: ModelTrainingOutput
    study: optuna.Study


class HyperparameterTuning:
    """Class for running an Optuna hyperparameter search for a given model type. Each trial trains a
    model with ModelTraining and scores it with ModelEvaluation on the primary_metric configured in
    config.yaml's hyperparameter_tuning section; the best-scoring trial's model and hyperparameters
    are returned. Run using dedicated orchestration script.
    """

    def __init__(
        self,
        model_trainer: ModelTraining,
        model_evaluator: ModelEvaluation,
        train_data: sp.csr_matrix,
        test_data: sp.csr_matrix,
        config: dict,
    ):
        """Loads and validates the hyperparameter_tuning section of the configuration.

        Args:
            model_trainer (ModelTraining): Trains a candidate model for a given hyperparameter set.
            model_evaluator (ModelEvaluation): Scores a candidate model against the held-out test split.
            train_data (sp.csr_matrix): The user-item matrix to train candidate models on.
            test_data (sp.csr_matrix): The held-out user-item matrix to evaluate candidate models on.
            config (dict): The full application config (as loaded from config.yaml)."""
        if "hyperparameter_tuning" not in config:
            raise ValueError("Missing 'hyperparameter_tuning' section in configuration.")

        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.train_data = train_data
        self.test_data = test_data

        tuning_config = config["hyperparameter_tuning"]
        self.n_trials = tuning_config.get("n_trials", 20)
        self.direction = tuning_config.get("direction", "maximize")
        self.primary_metric = tuning_config.get("primary_metric", "map_at_k")
        self.random_state = tuning_config.get("random_state", 42)
        self.search_spaces = tuning_config.get("search_space", {})

        if self.primary_metric not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown primary_metric '{self.primary_metric}'. Must be one of {sorted(METRIC_REGISTRY)}."
            )

    def _suggest_hyperparameters(self, trial: optuna.Trial, model_type: str) -> dict:
        model_key = model_type.lower()
        if model_key not in self.search_spaces:
            raise ValueError(f"No hyperparameter search space configured for model type '{model_type}'.")

        hyperparameters: dict[str, Any] = {}
        for param_name, spec in self.search_spaces[model_key].items():
            param_type = spec["type"]
            if param_type == "int":
                hyperparameters[param_name] = trial.suggest_int(
                    param_name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif param_type == "float":
                hyperparameters[param_name] = trial.suggest_float(
                    param_name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif param_type == "categorical":
                hyperparameters[param_name] = trial.suggest_categorical(param_name, spec["choices"])
            else:
                raise ValueError(
                    f"Unsupported search space type '{param_type}' for hyperparameter '{param_name}'. "
                    f"Must be one of {sorted(SUPPORTED_SEARCH_TYPES)}."
                )

        # random_state is held fixed across trials (not searched) so that trials only differ by the
        # parameters actually being tuned, keeping trial-to-trial comparisons fair.
        hyperparameters.setdefault("random_state", self.random_state)
        return hyperparameters

    def run(self, model_type: str) -> HyperparameterTuningOutput:
        """Runs the Optuna search for model_type and returns the best trial's model and hyperparameters.

        Args:
            model_type (str): The type of model to tune, either 'ALS' or 'BPR'.
        Returns:
            HyperparameterTuningOutput: The best hyperparameters, best score, best trained model, and
                the full Optuna study (for inspection/plotting)."""
        logger.info(
            f"Starting hyperparameter tuning for model type: {model_type} "
            f"({self.n_trials} trials, optimizing {self.primary_metric})"
        )

        trial_outputs: dict[int, ModelTrainingOutput] = {}

        def objective(trial: optuna.Trial) -> float:
            hyperparameters = self._suggest_hyperparameters(trial, model_type)
            training_output = self.model_trainer.train(model_type, hyperparameters)
            evaluation_output = self.model_evaluator.evaluate(training_output, self.train_data, self.test_data)

            trial_outputs[trial.number] = training_output
            for metric_name, score in evaluation_output.scores.items():
                trial.set_user_attr(metric_name, score)

            return evaluation_output.scores[self.primary_metric]

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(
            f"Hyperparameter tuning completed for model type: {model_type}. "
            f"Best {self.primary_metric}={study.best_value:.4f} with params: {study.best_params}"
        )

        return HyperparameterTuningOutput(
            model_type=model_type,
            best_hyperparameters=trial_outputs[study.best_trial.number].hyperparameters,
            best_score=study.best_value,
            best_training_output=trial_outputs[study.best_trial.number],
            study=study,
        )
