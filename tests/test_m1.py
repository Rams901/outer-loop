import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.harness import Budget, Harness
from rl_opt.m1 import m1_sim_filter
from rl_opt.simworld import SimWorld
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld


def test_label_reveals_v_short_does_not():
    world = TrueWorld.generate(
        WorldParams(n_users=80, n_posts=40, n_candidates=16, feed_size=6, horizon_days=5),
        seed=0,
    )
    h = Harness(world, Budget(80 * 5 + 80 * 2), np.random.default_rng(0))
    short = h.experiment(DEFAULT_WEIGHTS, n_users=80, n_days=2)
    assert short.value is None
    labeled = h.label(DEFAULT_WEIGHTS, n_users=80)
    assert labeled.value is not None
    assert 0.0 < labeled.value <= 1.0
    assert len(h.labeled()) == 1
    assert labeled.weights == tuple(float(x) for x in DEFAULT_WEIGHTS)


def test_simworld_ranks_default_above_engagement():
    world = TrueWorld.generate(
        WorldParams(n_users=120, n_posts=50, n_candidates=16, feed_size=6, horizon_days=6),
        seed=2,
    )
    h = Harness(world, Budget(120 * 6 * 4), np.random.default_rng(1))
    h.label(DEFAULT_WEIGHTS, 120)
    h.label(ENGAGEMENT_WEIGHTS, 120)
    w_mid = DEFAULT_WEIGHTS.copy()
    w_mid[0] = 1.2
    h.label(w_mid, 120)
    sim = SimWorld.fit(h.labeled())
    assert sim.predict_v(DEFAULT_WEIGHTS) > sim.predict_v(ENGAGEMENT_WEIGHTS)


def test_b5_only_ships_configs_it_measured():
    from rl_opt.search import b5_labels

    world = TrueWorld.generate(
        WorldParams(n_users=100, n_posts=40, n_candidates=16, feed_size=6, horizon_days=5),
        seed=4,
    )
    spec = PilotSpec(budget_user_days=100 * 5 * 6, m1_label_users=40)
    h = Harness(world, Budget(spec.budget_user_days), np.random.default_rng(3))
    commit = b5_labels(h, spec)
    labeled = h.labeled()
    assert len(labeled) >= 2
    assert all(r.value is not None for r in labeled)
    assert any(np.allclose(r.weights, commit.weights) for r in labeled)


def test_m1_commits_something():
    world = TrueWorld.generate(
        WorldParams(n_users=100, n_posts=40, n_candidates=16, feed_size=6, horizon_days=5),
        seed=3,
    )
    spec = PilotSpec(budget_user_days=100 * 5 * 10, m1_label_users=40, m1_confirm_users=40, m1_sim_draws=16)
    h = Harness(world, Budget(spec.budget_user_days), np.random.default_rng(2))
    commit = m1_sim_filter(h, spec)
    assert commit.weights.shape == (6,)
    assert commit.n_experiments >= 3
