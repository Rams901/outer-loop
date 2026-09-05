import numpy as np

from rl_opt.config import WorldParams
from rl_opt.world import TrueWorld


def test_base_rates_span_three_orders():
    world = TrueWorld.generate(WorldParams(n_users=400, n_posts=80), seed=0)
    rates = world.action_base_rates()
    assert rates["like"] / rates["report"] >= 500
    assert rates["report"] < 1e-3
    assert rates["like"] > 1e-2


def test_bait_inflates_likes_not_reports():
    world = TrueWorld.generate(WorldParams(n_users=400, n_posts=80), seed=1)
    p = world.true_probs()
    bait = world.catalogue.bait > 0.5
    like = p[:, :, 0]
    report = p[:, :, 5]
    assert like[:, bait].mean() > 1.15 * like[:, ~bait].mean()
    # Reports stay rare on bait; toxicity drives them, not emptiness.
    assert report[:, bait].mean() < 3.0 * report[:, ~bait].mean()


def test_rollout_emits_metrics():
    params = WorldParams(n_users=80, n_posts=40, n_candidates=16, feed_size=6, horizon_days=4)
    world = TrueWorld.generate(params, seed=2)
    from rl_opt.config import DEFAULT_WEIGHTS

    result = world.rollout(DEFAULT_WEIGHTS, seed=0)
    assert result.n_impressions > 0
    assert 0.0 < result.proxy < 1.0
    assert 0.0 < result.value <= 1.0
