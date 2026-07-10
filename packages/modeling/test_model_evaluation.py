import numpy as np
import pytest
import scipy.sparse as sp

from model_evaluation import METRIC_REGISTRY, ModelEvaluation, ModelEvaluationOutput
from model_training import ModelTraining


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
def training_output(train_data):
    config = {
        "model_training_als": {"factors": 2, "regularization": 0.01, "iterations": 2, "random_state": 42},
    }
    trainer = ModelTraining(train_data, {}, {}, config)
    return trainer.train("ALS")


@pytest.fixture
def config():
    return {"model_evaluation": {"k": 2, "metrics": ["precision_at_k", "map_at_k"]}}


def test_evaluate_returns_configured_metrics(training_output, train_data, test_data, config):
    evaluator = ModelEvaluation(config)

    result = evaluator.evaluate(training_output, train_data, test_data)

    assert isinstance(result, ModelEvaluationOutput)
    assert result.model_type == "ALS"
    assert result.hyperparameters == training_output.hyperparameters
    assert result.k == 2
    assert set(result.scores.keys()) == {"precision_at_k", "map_at_k"}
    for score in result.scores.values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_evaluate_defaults_to_all_metrics_when_not_configured(training_output, train_data, test_data):
    evaluator = ModelEvaluation({"model_evaluation": {}})

    result = evaluator.evaluate(training_output, train_data, test_data)

    assert set(result.scores.keys()) == set(METRIC_REGISTRY.keys())
    assert result.k == 10


def test_init_raises_for_missing_config_section():
    with pytest.raises(ValueError, match="Missing 'model_evaluation' section"):
        ModelEvaluation({})


def test_init_raises_for_unknown_metric():
    with pytest.raises(ValueError, match="Unknown evaluation metrics"):
        ModelEvaluation({"model_evaluation": {"metrics": ["recall_at_k"]}})
