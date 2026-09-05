import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.trace import pick_canonical_user, terms_sum_ok, trace_scenario
from rl_opt.world import TrueWorld


def test_score_terms_sum_to_total():
    world = TrueWorld.generate(WorldParams(n_users=200, n_posts=60, n_candidates=20, feed_size=8), seed=0)
    pick = pick_canonical_user(world, scenario_seed=11)
    trace = trace_scenario(world, pick.user_id, pick.candidate_ids)
    assert trace["schema"] == "rl_opt.trace.v1"
    assert all(terms_sum_ok(c) for c in trace["candidates"])


def test_canonical_user_disagrees_and_has_both_kinds():
    world = TrueWorld.generate(WorldParams(n_users=200, n_posts=60, n_candidates=20, feed_size=8), seed=0)
    pick = pick_canonical_user(world, scenario_seed=11)
    trace = trace_scenario(world, pick.user_id, pick.candidate_ids)
    baits = [c for c in trace["candidates"] if c["bait"] > 0.5]
    oks = [c for c in trace["candidates"] if c["bait"] <= 0.5]
    assert baits and oks
    top_d = trace["feed"]["default"]["slots"][0]
    top_e = trace["feed"]["engagement"]["slots"][0]
    assert top_d != top_e
    assert trace["feed"]["engagement"]["mean_bait"] > trace["feed"]["default"]["mean_bait"]


def test_terms_match_dot_product():
    from rl_opt.ranker import score_candidates

    world = TrueWorld.generate(WorldParams(n_users=80, n_posts=40, n_candidates=16, feed_size=6), seed=1)
    pick = pick_canonical_user(world, scenario_seed=3)
    trace = trace_scenario(world, pick.user_id, pick.candidate_ids)
    pred = world.predicted_probs()
    ids = np.array(pick.candidate_ids)
    for c, local in zip(trace["candidates"], range(len(ids))):
        p = pred[pick.user_id, ids[local]]
        assert np.isclose(c["score_terms"]["default"]["total"], float(score_candidates(p, DEFAULT_WEIGHTS)))
        assert np.isclose(c["score_terms"]["engagement"]["total"], float(score_candidates(p, ENGAGEMENT_WEIGHTS)))
