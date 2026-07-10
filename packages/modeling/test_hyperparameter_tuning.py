import numpy as np
import pytest
import scipy.sparse as sp

from hyperparameter_tuning import HyperparameterTuning, HyperparameterTuningOutput
from model_evaluation import ModelEvaluation
from model_training import ModelTraining, ModelTrainingOutput


@pytest.fixture
def train_data():
    return sp.csr_matrix(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def test_data():
    return sp.csr_matrix(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def config():
    return {
        "model_training_als": {"factors": 2, "regularization": 0.01, "iterations": 2, "random_state": 42},
        "model_evaluation": {"k": 2, "metrics": ["precision_at_k", "map_at_k"]},
        "hyperparameter_tuning": {
            "n_trials": 2,
            "direction": "maximize",
            "primary_metric": "map_at_k",
            "random_state": 42,
            "search_space": {
                "als": {
                    "factors": {"type": "int", "low": 2, "high": 3},
                    "regularization": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                    "iterations": {"type": "int", "low": 1, "high": 2},
                }
            },
        },
    }


@pytest.fixture
def tuner(train_data, test_data, config):
    trainer = ModelTraining(train_data, {}, {}, config)
    evaluator = ModelEvaluation(config)
    return HyperparameterTuning(trainer, evaluator, train_data, test_data, config)


def test_run_returns_best_trial_output(tuner):
    result = tuner.run("ALS")

    assert isinstance(result, HyperparameterTuningOutput)
    assert result.model_type == "ALS"
    assert isinstance(result.best_training_output, ModelTrainingOutput)
    assert result.best_hyperparameters == result.best_training_output.hyperparameters
    assert {"factors", "regularization", "iterations", "random_state"} <= result.best_hyperparameters.keys()
    assert result.best_hyperparameters["random_state"] == 42
    assert isinstance(result.best_score, float)
    assert len(result.study.trials) == 2


def test_init_raises_for_missing_config_section(train_data, test_data):
    trainer = ModelTraining(train_data, {}, {}, {"model_training_als": {}})
    evaluator = ModelEvaluation({"model_evaluation": {}})

    with pytest.raises(ValueError, match="Missing 'hyperparameter_tuning' section"):
        HyperparameterTuning(trainer, evaluator, train_data, test_data, {})


def test_run_raises_for_missing_search_space(tuner):
    tuner.search_spaces = {}

    with pytest.raises(ValueError, match="No hyperparameter search space configured"):
        tuner.run("ALS")


def test_init_raises_for_unknown_primary_metric(train_data, test_data, config):
    trainer = ModelTraining(train_data, {}, {}, config)
    evaluator = ModelEvaluation(config)
    config["hyperparameter_tuning"]["primary_metric"] = "recall_at_k"

    with pytest.raises(ValueError, match="Unknown primary_metric"):
        HyperparameterTuning(trainer, evaluator, train_data, test_data, config)
