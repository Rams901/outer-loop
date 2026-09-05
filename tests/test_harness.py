import numpy as np
import pytest

from rl_opt.config import DEFAULT_WEIGHTS, WorldParams
from rl_opt.harness import Budget, Harness, OverBudgetError, Oracle
from rl_opt.space import from_theta, to_theta
from rl_opt.world import TrueWorld


def test_from_theta_roundtrip_default():
    w = from_theta(to_theta(DEFAULT_WEIGHTS))
    assert np.allclose(w, DEFAULT_WEIGHTS, rtol=1e-6)


def test_budget_raises_and_does_not_overspend():
    b = Budget(10)
    b.charge(4)
    assert b.remaining == 6
    with pytest.raises(OverBudgetError):
        b.charge(7)
    assert b.remaining == 6


def test_harness_hides_v_on_short_experiments():
    world = TrueWorld.generate(
        WorldParams(n_users=80, n_posts=40, n_candidates=16, feed_size=6, horizon_days=6),
        seed=0,
    )
    h = Harness(world, Budget(80 * 2), np.random.default_rng(0))
    rec = h.experiment(DEFAULT_WEIGHTS, n_users=80, n_days=2)
    assert rec.value is None
    assert rec.proxy > 0
    assert rec.weights == tuple(float(x) for x in DEFAULT_WEIGHTS)


def test_oracle_returns_v():
    world = TrueWorld.generate(
        WorldParams(n_users=60, n_posts=30, n_candidates=12, feed_size=5, horizon_days=4),
        seed=1,
    )
    r = Oracle(world, rollout_seed=0).value(DEFAULT_WEIGHTS)
    assert 0.0 < r.value <= 1.0
