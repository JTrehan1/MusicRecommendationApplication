import numpy as np
import pytest
import scipy.sparse as sp

from model_training import ModelTraining, ModelTrainingOutput


@pytest.fixture
def csr_matrix():
    return sp.csr_matrix(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def config():
    return {
        "model_training_als": {
            "factors": 2,
            "regularization": 0.01,
            "iterations": 2,
            "random_state": 42,
        },
        "model_training_bpr": {
            "factors": 2,
            "regularization": 0.01,
            "iterations": 2,
            "learning_rate": 0.01,
            "random_state": 42,
        },
    }


@pytest.fixture
def trainer(csr_matrix, config):
    return ModelTraining(
        csr_matrix,
        track_uri_to_index={"track1": 0},
        index_to_track_uri={0: "track1"},
        config=config,
    )


def test_get_hyperparameters_returns_config_values(trainer, config):
    assert trainer.get_hyperparameters("ALS") == config["model_training_als"]
    assert trainer.get_hyperparameters("BPR") == config["model_training_bpr"]


def test_get_hyperparameters_raises_for_missing_config_section(csr_matrix):
    trainer = ModelTraining(csr_matrix, {}, {}, config={})

    with pytest.raises(ValueError, match="Missing 'model_training_als' section"):
        trainer.get_hyperparameters("ALS")


@pytest.mark.parametrize(
    "bad_hyperparameters, expected_match",
    [
        pytest.param(
            {"regularization": 0.01, "iterations": 2},
            "Missing required hyperparameters",
            id="missing_required_key",
        ),
        pytest.param(
            {"factors": 2, "regularization": 0.01, "iterations": 2, "not_a_real_param": 1},
            "Unexpected hyperparameters",
            id="unexpected_key",
        ),
    ],
)
def test_validate_hyperparameters_rejects_bad_config(trainer, bad_hyperparameters, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        trainer._validate_hyperparameters("ALS", bad_hyperparameters)


def test_train_raises_for_invalid_model_type(trainer):
    with pytest.raises(ValueError, match="Invalid model type specified"):
        trainer.train("KNN")


def test_train_uses_config_hyperparameters_by_default(trainer, config):
    result = trainer.train("ALS")

    assert isinstance(result, ModelTrainingOutput)
    assert result.model_type == "ALS"
    assert result.hyperparameters == config["model_training_als"]
    assert type(result.trained_model).__name__ == "AlternatingLeastSquares"
    assert result.track_uri_to_index == {"track1": 0}
    assert result.index_to_track_uri == {0: "track1"}


def test_train_uses_explicit_hyperparameters_over_config(trainer):
    explicit_hyperparameters = {"factors": 3, "regularization": 0.05, "iterations": 2, "random_state": 7}

    result = trainer.train("ALS", explicit_hyperparameters)

    assert result.hyperparameters == explicit_hyperparameters


def test_train_bpr_model_type(trainer, config):
    result = trainer.train("BPR")

    assert result.model_type == "BPR"
    assert type(result.trained_model).__name__ == "BayesianPersonalizedRanking"
    assert result.hyperparameters == config["model_training_bpr"]
