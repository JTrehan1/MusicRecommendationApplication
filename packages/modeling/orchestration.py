"""packages/modeling/orchestration.py - End-to-end modeling pipeline.

WHAT THIS SCRIPT DOES:
─────────────────────────────────────────
1. Loads config.yaml
2. Pulls playlist_tracks from Postgres
3. Cleans it (DataCleaning) and turns it into a playlist x track sparse matrix (DataIngestion)
4. Splits the matrix into train/test (DataSplitting)
5. Trains the model type configured in orchestration.model_type (ModelTraining), optionally running
   an Optuna hyperparameter search first (HyperparameterTuning) when orchestration.run_tuning is true
6. Scores the resulting model against the held-out test split (ModelEvaluation)
7. Persists the trained model, mappings, hyperparameters, and evaluation scores to artifacts.output_dir

RUN:
    python -m packages.modeling.orchestration
"""

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from packages.data_cleaning.data_cleaning import DataCleaning
from packages.data_ingestion.data_ingestion import DataIngestion, IngestionOutput
from packages.modeling.data_splitting import DataSplitting, DataSplittingOutput
from packages.modeling.hyperparameter_tuning import HyperparameterTuning
from packages.modeling.model_evaluation import ModelEvaluation, ModelEvaluationOutput
from packages.modeling.model_training import ModelTraining, ModelTrainingOutput
from pipelines.utils.db import get_conn

load_dotenv("credentials/.env")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_playlist_tracks() -> pd.DataFrame:
    log.info("Loading playlist_tracks from Postgres.")
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT playlist_id, track_uri, position FROM playlist_tracks", conn)
    log.info("Loaded %d playlist_tracks rows.", len(df))
    return df


def _prepare_data(config: dict) -> tuple[IngestionOutput, DataSplittingOutput]:
    playlist_tracks_df = _load_playlist_tracks()
    cleaned_df = DataCleaning(config["data"]).clean_data(playlist_tracks_df)
    ingestion_output = DataIngestion(cleaned_df).ingest()
    splitting_output = DataSplitting(ingestion_output).csr_splitter(**config["train_test_split"])
    return ingestion_output, splitting_output


def _train_model(
    config: dict,
    model_type: str,
    ingestion_output: IngestionOutput,
    splitting_output: DataSplittingOutput,
) -> ModelTrainingOutput:
    model_trainer = ModelTraining(
        splitting_output.train_data,
        ingestion_output.track_uri_to_index,
        ingestion_output.index_to_track_uri,
        config,
    )
    model_evaluator = ModelEvaluation(config)

    if config["orchestration"].get("run_tuning", False):
        tuning_output = HyperparameterTuning(
            model_trainer,
            model_evaluator,
            splitting_output.train_data,
            splitting_output.test_data,
            config,
        ).run(model_type)
        log.info("Selected best trial hyperparameters: %s", tuning_output.best_hyperparameters)
        return tuning_output.best_training_output

    return model_trainer.train(model_type)


def _persist_artifacts(
    config: dict,
    training_output: ModelTrainingOutput,
    evaluation_output: ModelEvaluationOutput,
) -> Path:
    output_dir = Path(config["artifacts"]["output_dir"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{training_output.model_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "model.pkl", "wb") as fh:
        pickle.dump(
            {
                "trained_model": training_output.trained_model,
                "track_uri_to_index": training_output.track_uri_to_index,
                "index_to_track_uri": training_output.index_to_track_uri,
            },
            fh,
        )

    metadata = {
        "model_type": training_output.model_type,
        "hyperparameters": training_output.hyperparameters,
        "evaluation": {"k": evaluation_output.k, "scores": evaluation_output.scores},
        "created_at": timestamp,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    log.info("Artifacts written to %s", run_dir)
    return run_dir


def run() -> Path:
    """Main entry point. Orchestrates the full modeling pipeline end-to-end and returns the
    directory the resulting artifacts were written to."""
    config = _load_config()
    model_type = config["orchestration"]["model_type"]

    ingestion_output, splitting_output = _prepare_data(config)
    training_output = _train_model(config, model_type, ingestion_output, splitting_output)

    # Re-score the final chosen model on the held-out split so the persisted metadata always
    # reflects the model actually being saved (rather than reusing scores recorded mid-search).
    evaluation_output = ModelEvaluation(config).evaluate(
        training_output, splitting_output.train_data, splitting_output.test_data
    )

    run_dir = _persist_artifacts(config, training_output, evaluation_output)
    log.info(
        "Pipeline complete for model type: %s. Evaluation scores: %s",
        model_type, evaluation_output.scores,
    )
    return run_dir


if __name__ == "__main__":
    run()
