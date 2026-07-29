# Popularity Baseline

## Why this exists
All ALS/BPR metrics (precision@k, MAP@k, NDCG@k, AUC@k) are currently uninterpretable
in isolation. Given the dataset's strong popularity bias (top 20% of tracks ≈ 63% of
interactions, from EDA), a naive "recommend the most popular tracks to everyone"
baseline may already score deceptively well. Until ALS is shown to beat this baseline
by a meaningful margin, tuning further is optimizing a number with no reference point.

## Interface contract (from `implicit.evaluation.ranking_metrics_at_k`)
- `model` must subclass `implicit.recommender_base.RecommenderBase` — an ABC.
- ABC requires 5 methods implemented: `fit`, `recommend`, `similar_users`,
  `similar_items`, `save`. Instantiation fails if any are missing.
- Only `recommend()` is actually called by `ranking_metrics_at_k`. The other four
  exist solely to satisfy the ABC.

## What `ranking_metrics_at_k` actually calls
```python
while start_idx < len(to_generate):
    batch = to_generate[start_idx : start_idx + batch_size]   # array of userids, up to 1000
    ids, _ = model.recommend(batch, train_user_items[np.asarray(batch)], N=K)
```
Key implications:
- `recommend()` is called in **batched form only** — `userid` is always an array,
  never a scalar. No need to handle a single-user case.
- `user_items` arg is a **CSR slice of multiple rows** (one row per user in batch),
  used for the `filter_already_liked_items` behavior.
- Return value `ids` must be **2D**: shape `(len(batch), N)`.
- The second return value (scores) is discarded (`_`) — metrics are computed purely
  from `ids` order. Real score values are not required, only correct ranking order.

## Class design: `PopularityBaseline(RecommenderBase)`

### `fit(self, user_items, show_progress=True, callback=None)`
- `user_items` = train CSR matrix (playlist × track).
- Sum interaction counts down the **item axis** → one popularity score per track.
- Sort descending, store as `self.popular_items` (item ids) and `self.popular_scores`.
- This is the only "training" — no iterative fitting needed.

### `recommend(self, userid, user_items, N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False, items=None)`
- Ignores per-user taste (same base ranking `self.popular_items` for everyone).
- If `filter_already_liked_items=True`: for each row in the batch, remove item ids
  already present in that user's `user_items` row before taking top-N.
  - Must match ALS's default filtering behavior for a fair, apples-to-apples
    comparison — otherwise the baseline could look artificially stronger or weaker.
  - Needs to be vectorized across the batch, not looped user-by-user in Python
    (batch can be up to 1000 rows — a naive Python loop reduces throughput
    but is functionally fine for a first pass; note as a possible optimization
    later if it's a bottleneck).
- Returns `(ids, scores)`:
  - `ids`: shape `(len(batch), N)`, top-N popular item ids per user after filtering.
  - `scores`: shape `(len(batch), N)` — content doesn't matter for these metrics
    (discarded by caller), but return something consistent (e.g. the popularity
    scores themselves) rather than garbage, in case it's reused elsewhere later.

### `fit`, `recommend` — real logic (above). Everything else — stub only:
```python
def similar_users(self, ...): raise NotImplementedError("Not supported for popularity baseline")
def similar_items(self, ...): raise NotImplementedError("Not supported for popularity baseline")
def save(self, file):         raise NotImplementedError("Not supported for popularity baseline")
```
Never called by `ranking_metrics_at_k` — exist only to satisfy the ABC.

## Usage (matches existing ALS evaluation call pattern — no changes needed elsewhere)
```python
baseline = PopularityBaseline()
baseline.fit(train_data)
scores = ranking_metrics_at_k(baseline, train_data, test_data, K=10)
```

## Open questions to resolve while implementing
1. Vectorizing the "remove already-liked items" step across a batched CSR slice —
   decide on an approach (e.g. build a boolean mask per row using `.indices`,
   or loop rows and use set differences) before assuming performance is fine at
   full dataset scale.
2. Confirm `self.popular_items` should exclude items with zero training interactions,
   or whether that's already guaranteed by upstream `min_track_occurrences` filtering
   in `DataCleaning`.
3. Decide where this class lives in the existing package structure — likely its own
   module (e.g. `packages/modeling/popularity_baseline.py`), kept separate from
   `ModelTraining` since it has no hyperparameters and should never enter the
   Optuna loop.

## Comparison step (after implementation)
- Run `PopularityBaseline` through the same `test_data`, same `k=10`, same 4 metrics.
- Place ALS's best scores side by side with baseline scores.
- Define what "beats baseline" means numerically (e.g. % relative lift) so there's
  a documented threshold rather than an eyeballed judgment call later.
