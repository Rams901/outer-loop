from rl_opt.config import WorldParams
from rl_opt.u1_strong import StrongSpec, u1_strong
from rl_opt.world import TrueWorld


def test_search_and_confirm_seeds_disjoint():
    spec = StrongSpec()
    assert set(spec.search_seeds).isdisjoint(spec.confirm_seeds)


def test_strong_u1_returns_headroom_on_tiny_world():
    spec = StrongSpec(
        search_seeds=(1,),
        confirm_seeds=(10, 11),
        n_random=3,
        es_generations=1,
        es_population=3,
        confirm_top_k=3,
    )
    world = TrueWorld.generate(
        WorldParams(n_users=80, n_posts=30, n_candidates=12, feed_size=5, horizon_days=4),
        seed=0,
    )
    res = u1_strong(world, 0, spec)
    assert res.n_calls > 0
    assert res.n_candidates >= 3
    assert 0.0 <= res.v0_confirm <= 1.0
    assert 0.0 <= res.v_star_confirm <= 1.0
    assert any(row["is_default"] for row in res.confirm_table)
