# Non-personalized baseline recommender, used to sanity-check that ALS/BPR actually beat
# "just recommend whatever's popular" on the same ranking metrics from implicit.evaluation.

import numpy as np
import scipy.sparse as sp
from implicit.recommender_base import RecommenderBase


class PopularityBaseline(RecommenderBase):
    """Recommends the same globally most-popular items to every user, ranked by total training
    interaction count. Only implements fit() and recommend() - the two methods
    implicit.evaluation's ranking_metrics_at_k actually calls.
    """

    def fit(self, user_items: sp.csr_matrix, show_progress: bool = True, callback=None):
        """Precomputes a fixed popularity ranking from the training interactions.

        Args:
            user_items (sp.csr_matrix): The user-item interaction matrix to fit on.
            show_progress (bool): Unused, present to match the RecommenderBase signature.
            callback: Unused, present to match the RecommenderBase signature."""
        item_scores = np.asarray(user_items.sum(axis=0)).ravel()
        order = np.argsort(-item_scores)

        self.popular_items = order
        self.popular_scores = item_scores[order]
        return self

    def recommend(
        self,
        userid,
        user_items: sp.csr_matrix,
        N: int = 10,
        filter_already_liked_items: bool = True,
        filter_items=None,
        recalculate_user: bool = False,
        items=None,
    ):
        """Returns the top-N most popular items for each user in userid, optionally filtering out
        items that user already has a training interaction with. Every user gets the same
        popularity ranking - userid is only used to index into user_items for filtering.

        Args:
            userid: Array of userids to recommend for (batched only, no scalar case).
            user_items (sp.csr_matrix): CSR slice with one row per userid, used to filter out
                already-liked items when filter_already_liked_items is True.
            N (int): Number of items to recommend per user.
            filter_already_liked_items (bool): When True, drop items already present in the
                user's user_items row, matching ALS/BPR's default recommend() behaviour.
            filter_items: Unused, present to match the RecommenderBase signature.
            recalculate_user (bool): Unused, present to match the RecommenderBase signature.
            items: Unused, present to match the RecommenderBase signature.
        Returns:
            tuple: (itemids, scores), both 2D arrays of shape (len(userid), N)."""
        userid = np.asarray(userid)
        batch_size = userid.shape[0]

        ids = np.empty((batch_size, N), dtype=self.popular_items.dtype)
        scores = np.empty((batch_size, N), dtype=self.popular_scores.dtype)

        for row in range(batch_size):
            candidate_items = self.popular_items
            candidate_scores = self.popular_scores

            if filter_already_liked_items:
                liked_items = user_items.indices[user_items.indptr[row]:user_items.indptr[row + 1]]
                if liked_items.size:
                    keep_mask = ~np.isin(candidate_items, liked_items)
                    candidate_items = candidate_items[keep_mask]
                    candidate_scores = candidate_scores[keep_mask]

            ids[row] = candidate_items[:N]
            scores[row] = candidate_scores[:N]

        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError("Not supported for popularity baseline")

    def similar_items(self, itemid, N=10, recalculate_item=False, item_users=None,
                       filter_items=None, items=None):
        raise NotImplementedError("Not supported for popularity baseline")

    def save(self, file):
        raise NotImplementedError("Not supported for popularity baseline")
