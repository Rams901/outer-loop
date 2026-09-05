import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.fidelity import FidelitySim
from rl_opt.prove_c4 import C4_THRESHOLD, MIN_SEGMENT, evaluate_seed
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld


def _small_params(**kwargs) -> WorldParams:
    base = dict(
        n_users=120,
        n_posts=40,
        n_candidates=16,
        feed_size=6,
        horizon_days=4,
        n_topics=2,
    )
    base.update(kwargs)
    return WorldParams(**base)


def test_rotated_interests_changes_relevance():
    world = TrueWorld.generate(_small_params(), seed=0)
    stale = world.with_rotated_interests(1)
    assert not np.allclose(world.relevance(), stale.relevance())
    assert np.allclose(world.catalogue.quality, stale.catalogue.quality)


def test_rare_pred_scale_changes_hide_report_predictions():
    world = TrueWorld.generate(_small_params(), seed=1)
    rare = world.with_rare_pred_scale(5.0)
    pred = world.predicted_probs()
    pred_r = rare.predicted_probs()
    assert not np.allclose(pred[:, :, 4], pred_r[:, :, 4])
    assert not np.allclose(pred[:, :, 5], pred_r[:, :, 5])
    assert np.allclose(world.true_probs(), rare.true_probs())


def test_proxy_axis_at_zero_prefers_high_proxy_config():
    world = TrueWorld.generate(_small_params(), seed=2)
    sim = FidelitySim(world, phi=0.0, n_users=80, rollout_seed=1, axis="proxy")
    # Engagement is the short-horizon trap; default is not.
    assert sim.predict_v(ENGAGEMENT_WEIGHTS) >= sim.predict_v(DEFAULT_WEIGHTS) - 1e-9


def test_c4_runs_on_tiny_world():
    spec = PilotSpec(u1_random=0)
    row = evaluate_seed(0, spec, _small_params(n_users=160, n_topics=2))
    assert "gap_over_headroom" in row
    assert isinstance(row["passed"], bool)
    assert row["headroom"] == row["v_star_global"] - row["v0"]
    assert C4_THRESHOLD == 0.20
    assert MIN_SEGMENT >= 1
