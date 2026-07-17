# Model evaluation script which scores a trained ALS/BPR model against held-out test interactions.

from dataclasses import dataclass
import logging
import scipy.sparse as sp
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k, AUC_at_k

from packages.modeling.model_training import ModelTrainingOutput

logger = logging.getLogger(__name__)

# implicit's ranking metrics all share the signature (model, train_user_items, test_user_items, K, ...).
METRIC_REGISTRY = {
    "precision_at_k": precision_at_k,
    "map_at_k": mean_average_precision_at_k,
    "ndcg_at_k": ndcg_at_k,
    "auc_at_k": AUC_at_k,
}


@dataclass
class ModelEvaluationOutput:
    model_type: str
    hyperparameters: dict
    k: int
    scores: dict


class ModelEvaluation:
    """Class for evaluating trained ALS/BPR models against a held-out test split, using the
    ranking metrics from implicit.evaluation. Metrics and k are sourced from config.yaml.
    """

    def __init__(self, config: dict):
        """Loads and validates the model_evaluation section of the configuration.

        Args:
            config (dict): The full application config (as loaded from config.yaml)."""
        if "model_evaluation" not in config:
            raise ValueError("Missing 'model_evaluation' section in configuration.")

        evaluation_config = config["model_evaluation"]
        self.k = evaluation_config.get("k", 10)
        self.metric_names = evaluation_config.get("metrics", list(METRIC_REGISTRY))

        unknown_metrics = set(self.metric_names) - METRIC_REGISTRY.keys()
        if unknown_metrics:
            raise ValueError(
                f"Unknown evaluation metrics in config: {sorted(unknown_metrics)}. "
                f"Must be one of {sorted(METRIC_REGISTRY)}."
            )

    def evaluate(
        self,
        training_output: ModelTrainingOutput,
        train_data: sp.csr_matrix,
        test_data: sp.csr_matrix,
    ) -> ModelEvaluationOutput:
        """Scores a trained model against the held-out test interactions for every configured metric.

        Args:
            training_output (ModelTrainingOutput): The output of ModelTraining.train().
            train_data (sp.csr_matrix): The user-item matrix the model was trained on.
            test_data (sp.csr_matrix): The held-out user-item matrix to evaluate against, matching
                train_data's shape (as produced by DataSplitting.csr_splitter()).
        Returns:
            ModelEvaluationOutput: A dataclass containing the model type, hyperparameters, k, and
                the computed evaluation scores."""
        logger.info(
            f"Evaluating {training_output.model_type} model at k={self.k} using metrics: {self.metric_names}"
        )

        scores = {}
        for metric_name in self.metric_names:
            metric_fn = METRIC_REGISTRY[metric_name]
            scores[metric_name] = float(
                metric_fn(training_output.trained_model, train_data, test_data, K=self.k, show_progress=False)
            )

        logger.info(f"Evaluation scores for {training_output.model_type}: {scores}")

        return ModelEvaluationOutput(
            model_type=training_output.model_type,
            hyperparameters=training_output.hyperparameters,
            k=self.k,
            scores=scores,
        )
