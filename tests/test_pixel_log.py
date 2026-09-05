import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, WorldParams
from rl_opt.pixel_log import pick_watch_ids
from rl_opt.world import TrueWorld


def test_watch_log_tracks_days_and_feeds():
    params = WorldParams(n_users=60, n_posts=30, n_candidates=12, feed_size=4, horizon_days=5)
    world = TrueWorld.generate(params, seed=0)
    watch = np.array([0, 1, 2], dtype=np.int64)
    result = world.rollout(DEFAULT_WEIGHTS, seed=0, watch_ids=watch)
    log = world.last_pixel_log
    assert log is not None
    assert log["schema"] == "rl_opt.pixel.v1"
    assert len(log["days"]) == 5
    assert log["days"][0][0]["returned"] is True
    assert result.value > 0
    fed = next(r for r in log["days"][0] if r["returned"])
    assert len(fed["feed"]) == 4
    assert "in_interest" in fed["feed"][0]


def test_pick_watch_stratifies_topics():
    world = TrueWorld.generate(WorldParams(n_users=80, n_posts=40, n_topics=4), seed=1)
    ids = pick_watch_ids(world, 16, seed=2)
    assert ids.size == 16
    topics = world.primary_topic()[ids]
    assert len(set(topics.tolist())) >= 2
