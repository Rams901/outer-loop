"""Step 0 diagnostic: is `V* - V0` real, or an artifact of a weak oracle?

The phase-1 U1 made 30 oracle calls at a single rollout seed, and on 4 of 5
seeds its answer was `QUALITY_WEIGHTS` -- a config we wrote by hand and placed
in its own candidate pool. Measured rollout noise on V is sd ~0.002 per call,
the same order as the 0.002-0.013 headroom it reported, so best-of-30 selection
bias alone could account for the number.

This module searches properly and, critically, confirms on rollout seeds that
were not used during the search, so the reported optimum is an out-of-sample
estimate rather than the maximum of the noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS
from rl_opt.metrics import safety_violated
from rl_opt.oracle_search import QUALITY_WEIGHTS, _axis_neighbours
from rl_opt.space import LOG_HI, LOG_LO, from_theta, sample_theta, to_theta
from rl_opt.world import TrueWorld


@dataclass(frozen=True)
class StrongSpec:
    """Search seeds and confirm seeds are disjoint. That is the whole point."""

    search_seeds: tuple[int, ...] = (100, 101)
    confirm_seeds: tuple[int, ...] = tuple(range(200, 208))
    n_random: int = 32
    es_generations: int = 8
    es_population: int = 12
    es_sigma0: float = 0.55
    es_sigma_decay: float = 0.85
    confirm_top_k: int = 12


@dataclass
class Candidate:
    weights: NDArray[np.float64]
    search_value: float
    unsafe: bool


@dataclass
class StrongResult:
    world_seed: int
    v0_confirm: float
    v0_sd: float
    v_star_confirm: float
    v_star_sd: float
    w_star: list[float]
    headroom: float
    headroom_se: float
    n_calls: int
    n_candidates: int
    search_curve: list[float] = field(default_factory=list)
    confirm_table: list[dict] = field(default_factory=list)
    w_star_is_planted: bool = False


class _Counter:
    def __init__(self) -> None:
        self.n = 0


def _evaluate(
    world: TrueWorld, weights: NDArray[np.float64], seeds: tuple[int, ...], counter: _Counter
) -> tuple[float, float, bool]:
    vals, hides, reports = [], [], []
    for s in seeds:
        r = world.rollout(weights, seed=s)
        counter.n += 1
        vals.append(r.value)
        hides.append(r.hide_rate)
        reports.append(r.report_rate)
    arr = np.array(vals)
    unsafe = safety_violated(float(np.mean(hides)), float(np.mean(reports)), world.params)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), sd, unsafe


def _seed_pool(rng: np.random.Generator, spec: StrongSpec) -> list[NDArray[np.float64]]:
    pool = [DEFAULT_WEIGHTS.copy(), ENGAGEMENT_WEIGHTS.copy(), QUALITY_WEIGHTS.copy()]
    pool.extend(_axis_neighbours())
    pool.extend(from_theta(t) for t in sample_theta(rng, spec.n_random))
    return pool


def u1_strong(world: TrueWorld, world_seed: int, spec: StrongSpec | None = None) -> StrongResult:
    spec = spec or StrongSpec()
    rng = np.random.default_rng(world_seed * 7919 + 5)
    counter = _Counter()
    seen: list[Candidate] = []
    curve: list[float] = []
    best_so_far = -np.inf

    def probe(w: NDArray[np.float64]) -> float:
        nonlocal best_so_far
        v, _, unsafe = _evaluate(world, w, spec.search_seeds, counter)
        seen.append(Candidate(np.asarray(w, dtype=np.float64).copy(), v, unsafe))
        if not unsafe and v > best_so_far:
            best_so_far = v
        curve.append(best_so_far)
        return v

    for w in _seed_pool(rng, spec):
        probe(w)

    safe = [c for c in seen if not c.unsafe]
    start = max(safe, key=lambda c: c.search_value) if safe else seen[0]
    mean = to_theta(start.weights)
    sigma = spec.es_sigma0
    for _ in range(spec.es_generations):
        thetas = np.clip(mean + sigma * rng.normal(size=(spec.es_population, mean.size)), LOG_LO, LOG_HI)
        scored = [(probe(from_theta(t)), t) for t in thetas]
        scored = [(v, t) for v, t in scored if np.isfinite(v)]
        scored.sort(key=lambda x: x[0], reverse=True)
        parents = np.stack([t for _, t in scored[: max(1, len(scored) // 2)]])
        mean = parents.mean(axis=0)
        sigma = max(0.06, spec.es_sigma_decay * sigma)

    # Confirm on held-out rollout seeds. Default is always in the pool so V0 and
    # V* are measured by the identical protocol.
    safe = [c for c in seen if not c.unsafe]
    safe.sort(key=lambda c: c.search_value, reverse=True)
    finalists = safe[: spec.confirm_top_k]
    if not any(np.allclose(c.weights, DEFAULT_WEIGHTS) for c in finalists):
        finalists.append(Candidate(DEFAULT_WEIGHTS.copy(), float("nan"), False))

    table = []
    for c in finalists:
        v, sd, unsafe = _evaluate(world, c.weights, spec.confirm_seeds, counter)
        table.append(
            {
                "weights": [float(x) for x in c.weights],
                "search_value": c.search_value,
                "confirm_value": v,
                "confirm_sd": sd,
                "unsafe": unsafe,
                "is_default": bool(np.allclose(c.weights, DEFAULT_WEIGHTS)),
                "is_quality": bool(np.allclose(c.weights, QUALITY_WEIGHTS)),
            }
        )

    default_row = next(r for r in table if r["is_default"])
    best_row = max((r for r in table if not r["unsafe"]), key=lambda r: r["confirm_value"])
    n_conf = len(spec.confirm_seeds)
    se = float(np.sqrt(default_row["confirm_sd"] ** 2 + best_row["confirm_sd"] ** 2) / np.sqrt(n_conf))
    table.sort(key=lambda r: r["confirm_value"], reverse=True)

    return StrongResult(
        world_seed=world_seed,
        v0_confirm=default_row["confirm_value"],
        v0_sd=default_row["confirm_sd"],
        v_star_confirm=best_row["confirm_value"],
        v_star_sd=best_row["confirm_sd"],
        w_star=best_row["weights"],
        headroom=best_row["confirm_value"] - default_row["confirm_value"],
        headroom_se=se,
        n_calls=counter.n,
        n_candidates=len(seen),
        search_curve=curve,
        confirm_table=table,
        w_star_is_planted=bool(best_row["is_quality"]),
    )
