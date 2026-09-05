import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, N_ACTIONS
from rl_opt.ranker import rank_indices, score_candidates


def test_scale_symmetry():
    rng = np.random.default_rng(0)
    pred = rng.random((32, 20, N_ACTIONS))
    a = rank_indices(pred, DEFAULT_WEIGHTS, k=8)
    b = rank_indices(pred, DEFAULT_WEIGHTS * 3.7, k=8)
    assert np.array_equal(a, b)


def test_higher_like_weight_promotes_likeable_items():
    pred = np.zeros((1, 2, N_ACTIONS))
    pred[0, 0, 0] = 0.9  # item 0: high like
    pred[0, 1, 1] = 0.9  # item 1: high reply
    like_heavy = np.array([10.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    reply_heavy = np.array([0.1, 10.0, 0.0, 0.0, 0.0, 0.0])
    assert rank_indices(pred, like_heavy, k=1)[0, 0] == 0
    assert rank_indices(pred, reply_heavy, k=1)[0, 0] == 1


def test_score_is_dot_product():
    p = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.01])
    w = DEFAULT_WEIGHTS
    assert np.isclose(score_candidates(p, w), float(p @ w))
