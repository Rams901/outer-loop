from rl_opt.stage2 import PILOT_JSON, build_lock
from rl_opt.stats import N_CEILING, N_FLOOR


def test_stage2_n_from_pilot_and_avoids_used_seeds():
    lock = build_lock(PILOT_JSON)
    assert N_FLOOR <= lock.n <= N_CEILING
    assert lock.confirmatory_seeds[0] == 30
    assert len(lock.confirmatory_seeds) == lock.n
    used = set(lock.pilot_seeds) | set(lock.development_seeds)
    assert used.isdisjoint(lock.confirmatory_seeds)
    assert lock.frozen_m1 == "m1_gp_lcb_one"
    assert lock.l_labels >= 2
