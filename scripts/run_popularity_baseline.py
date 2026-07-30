"""scripts/run_popularity_baseline.py - Score the non-personalized popularity baseline.

WHAT THIS SCRIPT DOES:
─────────────────────────────────────────
Runs PopularityBaseline through the *exact same* train/test split and ranking metrics that
orchestration.py evaluates ALS/BPR on, so the baseline scores can be eyeballed side by side
with the ALS artifact metadata.json without worrying about the comparison silently drifting.

1. Reuses orchestration._prepare_data (DataCleaning -> DataIngestion -> DataSplitting), driven by
   the same config.yaml sections ALS uses (data, train_test_split) - not reimplemented here.
2. Fits PopularityBaseline on the train split.
3. Calls implicit.evaluation.ranking_metrics_at_k(baseline, train, test, K=k) directly, with k
   sourced from config.model_evaluation.k (the same k ALS is scored at).
4. Prints/logs the result in the same shape as the ALS artifact metadata.json.

RUN (from the repo root):
    python scripts/run_popularity_baseline.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `packages` / `pipelines` importable when this file is run directly as a script
# (python scripts/run_popularity_baseline.py) rather than as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implicit.evaluation import ranking_metrics_at_k

# Reuse orchestration's data-loading pipeline verbatim so the baseline sees the identical split
# ALS was evaluated on, built from the same config-driven cleaning/splitting parameters.
from packages.modeling.orchestration import _load_config, _prepare_data
from packages.modeling.popularity_baseline import PopularityBaseline

log = logging.getLogger(__name__)

# ranking_metrics_at_k returns short keys; map them onto the names ModelEvaluation writes into the
# ALS artifact's evaluation.scores so both JSON blobs line up key-for-key.
METRIC_KEY_MAP = {
    "precision": "precision_at_k",
    "map": "map_at_k",
    "ndcg": "ndcg_at_k",
    "auc": "auc_at_k",
}


def run() -> dict:
    """Fits and scores the popularity baseline, returning an ALS-artifact-shaped result dict."""
    config = _load_config()
    k = config.get("model_evaluation", {}).get("k", 10)

    # Same DataCleaning -> DataIngestion -> DataSplitting path (and same config params) as ALS.
    _ingestion_output, splitting_output = _prepare_data(config)
    train_data = splitting_output.train_data
    test_data = splitting_output.test_data

    baseline = PopularityBaseline()
    baseline.fit(train_data)

    log.info("Scoring PopularityBaseline at k=%d via ranking_metrics_at_k.", k)
    raw_scores = ranking_metrics_at_k(baseline, train_data, test_data, K=k, show_progress=False)
    scores = {METRIC_KEY_MAP.get(name, name): float(value) for name, value in raw_scores.items()}

    result = {
        "model_type": "PopularityBaseline",
        "hyperparameters": {},  # baseline has none - present to match the ALS artifact shape.
        "evaluation": {"k": k, "scores": scores},
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }

    log.info("PopularityBaseline evaluation scores: %s", scores)
    # Print the artifact-shaped JSON to stdout so it's trivial to diff against an ALS metadata.json.
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
