"""Wilcoxon, bootstrap CI, and the stage-2 sample-size formula.

No scipy. Floor 20 / ceiling 200 / MDE 0.10 / 80% power / α = 0.05 are the
pre-registered constants; this module only executes them.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Z_ALPHA_2 = 1.959963984540054  # Φ^{-1}(0.975)
Z_POWER_80 = 0.841621233572914  # Φ^{-1}(0.80)
WILCOXON_ARE = 3.0 / math.pi  # vs paired t, under normality
MDE = 0.10
ALPHA = 0.05
POWER = 0.80
N_FLOOR = 20
N_CEILING = 200


def paired_sd(diffs: NDArray[np.float64] | list[float]) -> float:
    x = np.asarray(diffs, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least two paired differences for a variance")
    return float(np.std(x, ddof=1))


def sample_size_from_pilot_sd(sd: float, mde: float = MDE) -> dict:
    """Paired-t n, then Wilcoxon inflation, then floor/ceiling.

    The pilot n=5 sd is noisy (one exploitation seed dominates). The protocol
    still uses it rather than a judgement call.
    """
    if sd <= 0 or not math.isfinite(sd):
        n_t = N_FLOOR
    else:
        n_t = math.ceil((Z_ALPHA_2 + Z_POWER_80) ** 2 * (sd**2) / (mde**2))
    n_w = math.ceil(n_t / WILCOXON_ARE)
    n = int(min(N_CEILING, max(N_FLOOR, n_w)))
    return {
        "pilot_sd": float(sd),
        "mde": float(mde),
        "alpha": ALPHA,
        "power": POWER,
        "n_paired_t": int(n_t),
        "n_wilcoxon": int(n_w),
        "n": n,
        "floor": N_FLOOR,
        "ceiling": N_CEILING,
    }


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j + 1)
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: NDArray[np.float64] | list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank on paired differences.

    Zero differences are dropped. P-value is the normal approximation with
    tie correction; exact tables are not used above n≈20.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.abs(d) > 1e-15]
    n = int(d.size)
    if n < 1:
        return {"n": 0, "w_plus": 0.0, "z": 0.0, "p_value": 1.0, "median": 0.0}
    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    expected = n * (n + 1) / 4.0
    # Tie correction on absolute values.
    var = n * (n + 1) * (2 * n + 1) / 24.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = np.sum(counts**3 - counts)
    var -= tie / 48.0
    var = max(var, 1e-12)
    z = (w_plus - expected) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "n": n,
        "n_dropped_zeros": int(np.asarray(diffs).size - n),
        "w_plus": w_plus,
        "z": float(z),
        "p_value": float(p),
        "median": float(np.median(d)),
    }


def bootstrap_median_ci(
    diffs: NDArray[np.float64] | list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    x = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    meds = np.empty(n_resamples, dtype=np.float64)
    n = x.size
    for i in range(n_resamples):
        meds[i] = float(np.median(rng.choice(x, size=n, replace=True)))
    lo = float(np.quantile(meds, alpha / 2))
    hi = float(np.quantile(meds, 1.0 - alpha / 2))
    return {
        "median": float(np.median(x)),
        "ci_low": lo,
        "ci_high": hi,
        "n_resamples": n_resamples,
        "alpha": alpha,
    }
