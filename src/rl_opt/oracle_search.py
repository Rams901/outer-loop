"""U1: unlimited-budget search on true V. Computes w* for regret."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS
from rl_opt.harness import Oracle
from rl_opt.space import LOG_HI, LOG_LO, PilotSpec, from_theta, sample_theta, to_theta

QUALITY_WEIGHTS = np.array([0.15, 8.0, 0.02, 8.0, -70.0, -300.0], dtype=np.float64)


def _axis_neighbours() -> list[NDArray[np.float64]]:
    out: list[NDArray[np.float64]] = []
    grids = [
        [0.15, 0.8, 1.8, 2.6],
        [3.5, 8.0],
        [0.02, 0.55, 1.4],
        [1.0, 4.0, 8.0],
        [-43.2, -70.0],
        [-234.0, -320.0],
    ]
    for i, vals in enumerate(grids):
        for v in vals:
            w = DEFAULT_WEIGHTS.copy()
            w[i] = v
            out.append(w)
    return out


def u1_oracle(oracle: Oracle, spec: PilotSpec) -> tuple[NDArray[np.float64], float, float]:
    """Brute-force optimum on true V.

    U1 is allowed to know things entrants do not -- it is the oracle, not a
    competitor. It gets the hand-built quality direction *and* an ES refinement
    stage, because a `V*` that an entrant can beat makes regret negative and
    meaningless. That happened with the old 30-call version.
    """
    rng = np.random.default_rng(spec.seed0 + 3)
    pool = [DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, QUALITY_WEIGHTS]
    pool.extend(_axis_neighbours())
    pool.extend(from_theta(sample_theta(rng)[0]) for _ in range(spec.u1_random))
    best_w = DEFAULT_WEIGHTS.copy()
    best_v = -np.inf
    v0 = None
    for w in pool:
        r = oracle.value(w)
        if np.allclose(w, DEFAULT_WEIGHTS):
            v0 = r.value
        if r.unsafe:
            continue
        if r.value > best_v:
            best_v = r.value
            best_w = w.copy()
    if v0 is None:
        v0 = oracle.value(DEFAULT_WEIGHTS).value

    mean = to_theta(best_w)
    sigma = spec.u1_es_sigma0
    for _ in range(spec.u1_es_generations):
        thetas = np.clip(
            mean + sigma * rng.normal(size=(spec.u1_es_population, mean.size)), LOG_LO, LOG_HI
        )
        scored: list[tuple[float, NDArray[np.float64]]] = []
        for theta in thetas:
            r = oracle.value(from_theta(theta))
            if r.unsafe:
                continue
            scored.append((r.value, theta))
            if r.value > best_v:
                best_v = r.value
                best_w = from_theta(theta)
        if not scored:
            break
        scored.sort(key=lambda t: t[0], reverse=True)
        parents = np.stack([t[1] for t in scored[: max(1, len(scored) // 2)]])
        mean = parents.mean(axis=0)
        sigma = max(0.07, spec.u1_es_decay * sigma)
    return best_w, float(best_v), float(v0)
