import math

import numpy as np

from rl_opt.stats import (
    bootstrap_median_ci,
    sample_size_from_pilot_sd,
    wilcoxon_signed_rank,
)


def test_sample_size_respects_floor_and_ceiling():
    low = sample_size_from_pilot_sd(0.01)
    assert low["n"] == 20
    high = sample_size_from_pilot_sd(10.0)
    assert high["n"] == 200
    mid = sample_size_from_pilot_sd(0.24)
    assert 20 <= mid["n"] <= 200
    assert mid["n_wilcoxon"] >= mid["n_paired_t"]


def test_wilcoxon_detects_shift():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.2, 0.05, size=30)
    out = wilcoxon_signed_rank(diffs)
    assert out["p_value"] < 0.05
    assert out["median"] > 0


def test_wilcoxon_symmetric_not_significant():
    diffs = np.array([-0.2, -0.1, 0.1, 0.2, -0.05, 0.05])
    out = wilcoxon_signed_rank(diffs)
    assert out["p_value"] > 0.2


def test_bootstrap_ci_covers_median():
    diffs = np.linspace(-0.04, 0.04, 21)
    ci = bootstrap_median_ci(diffs, n_resamples=2000, seed=2)
    assert ci["ci_low"] <= ci["median"] <= ci["ci_high"]
    assert math.isfinite(ci["ci_low"])
