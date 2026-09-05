"""Inner-loop ranker: weighted combination of predicted action probabilities.

Mirrors the structural fact in `compute_weighted_score` — score is a linear
combination of predicted probabilities — without copying the offset/dwell-regret
machinery. Ranking of `w` and `k*w` for k>0 is identical; that is the scale
symmetry Q12 asked us to verify before searching on a sphere.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def score_candidates(predictions: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """predictions: (..., n_actions), weights: (n_actions,) -> (...,) scores."""
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("weights must be a 1-d vector")
    if p.shape[-1] != w.shape[0]:
        raise ValueError(f"last axis {p.shape[-1]} does not match weights {w.shape[0]}")
    return p @ w


def rank_indices(predictions: NDArray[np.float64], weights: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Top-k indices along axis 1. predictions: (n_users, n_cand, n_actions)."""
    scores = score_candidates(predictions, weights)
    # argpartition then sort the shortlist so slot 0 is the best.
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(-top_scores, axis=1)
    return np.take_along_axis(top, order, axis=1)


def rankings_equal(a: NDArray[np.int64], b: NDArray[np.int64]) -> bool:
    return bool(np.array_equal(a, b))
