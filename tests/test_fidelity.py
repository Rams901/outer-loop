import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.fidelity import FidelitySim, m1_against_sim, sim_score
from rl_opt.harness import Budget, Harness
from rl_opt.run_fidelity import _crossover
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld


def test_sim_score_is_v_at_phi_one_and_bait_at_zero():
    assert sim_score(0.6, 0.2, 1.0) == 0.6
    assert sim_score(0.6, 0.2, 0.0) == 0.2
    assert abs(sim_score(0.6, 0.2, 0.5) - 0.4) < 1e-12


def test_phi_one_prefers_default_over_engagement():
    world = TrueWorld.generate(
        WorldParams(n_users=80, n_posts=40, n_candidates=16, feed_size=6, horizon_days=5),
        seed=0,
    )
    sim = FidelitySim(world, phi=1.0, n_users=80, rollout_seed=1)
    assert sim.predict_v(DEFAULT_WEIGHTS) > sim.predict_v(ENGAGEMENT_WEIGHTS)


def test_phi_zero_prefers_engagement_over_default():
    world = TrueWorld.generate(
        WorldParams(n_users=80, n_posts=40, n_candidates=16, feed_size=6, horizon_days=5),
        seed=0,
    )
    sim = FidelitySim(world, phi=0.0, n_users=80, rollout_seed=1)
    assert sim.predict_v(ENGAGEMENT_WEIGHTS) > sim.predict_v(DEFAULT_WEIGHTS)


def test_m1_against_sim_commits_and_charges_labels():
    world = TrueWorld.generate(
        WorldParams(n_users=80, n_posts=40, n_candidates=16, feed_size=6, horizon_days=4),
        seed=1,
    )
    spec = PilotSpec(budget_user_days=80 * 4 * 6, m1_label_users=40, m1_sim_draws=6)
    sim = FidelitySim(world, phi=1.0, n_users=40, rollout_seed=2)
    harness = Harness(world, Budget(spec.budget_user_days), np.random.default_rng(0))
    commit = m1_against_sim(harness, spec, sim)
    assert commit.weights.shape == (6,)
    assert 1 <= commit.n_experiments <= 2
    assert len(harness.labeled()) == commit.n_experiments


def test_crossover_interpolates():
    assert _crossover([0.0, 0.5, 1.0], [0.4, 0.5, 0.7], 0.6) == 0.75
    assert _crossover([0.0, 1.0], [0.7, 0.8], 0.6) == 0.0
    assert _crossover([0.0, 1.0], [0.4, 0.5], 0.6) is None
