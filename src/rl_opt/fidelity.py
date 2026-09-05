"""Hand-specified simulator with a scalar fidelity dial (Q9).

φ = 1 maximises true delayed V. φ = 0 maximises bait share — the missing-feature
failure in Q8, where the sim cannot see that bait hurts retention. Intermediate
values mix them. Queries are free and use a user subset so the sweep fits on a
laptop.

This is not the fitted GP from `m1_sim_filter`. That method already lost to B5
at one implicit fidelity. The curve here answers the upper-bound question:
how correct does a simulator have to be before searching it beats buying
eleven honest labels?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS
from rl_opt.harness import Harness, OverBudgetError
from rl_opt.search import Commit
from rl_opt.space import LOG_HI, LOG_LO, PilotSpec, from_theta, sample_theta, to_theta
from rl_opt.world import TrueWorld

PHI_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
Q8_PHI_GRID = (0.0, 0.5, 1.0)
Q8_AXES = ("bait", "proxy", "stale", "rare")


def sim_score(value: float, bait_share: float, phi: float) -> float:
    """Scalar objective the agent maximises in sim. Monotone in φ."""
    phi = float(np.clip(phi, 0.0, 1.0))
    return phi * float(value) + (1.0 - phi) * float(bait_share)


@dataclass
class FidelitySim:
    """φ=1 true V. φ=0 missing-feature: the sim maximises bait share."""

    world: TrueWorld
    phi: float
    n_users: int = 400
    rollout_seed: int = 17
    axis: str = "bait"
    user_ids: NDArray[np.int64] = field(init=False)
    _alt: TrueWorld | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        n = min(int(self.n_users), self.world.params.n_users)
        rng = np.random.default_rng(self.rollout_seed + 99)
        self.user_ids = rng.choice(self.world.params.n_users, size=n, replace=False)
        if self.axis == "stale":
            self._alt = self.world.with_rotated_interests(1)
        elif self.axis == "rare":
            self._alt = self.world.with_rare_pred_scale(5.0)
        else:
            self._alt = None

    @property
    def tag(self) -> str:
        return f"fidelity_{self.axis}_phi_{self.phi:g}"

    def query(self, weights: NDArray[np.float64]) -> tuple[float, float, float]:
        result = self.world.rollout(
            weights,
            n_days=self.world.params.horizon_days,
            seed=self.rollout_seed,
            user_ids=self.user_ids,
        )
        phi = float(np.clip(self.phi, 0.0, 1.0))
        if self.axis == "bait":
            wrong = float(result.bait_share)
        elif self.axis == "proxy":
            wrong = float(result.proxy)
        elif self.axis in {"stale", "rare"}:
            assert self._alt is not None
            wrong = float(
                self._alt.rollout(
                    weights,
                    n_days=self.world.params.horizon_days,
                    seed=self.rollout_seed,
                    user_ids=self.user_ids,
                ).value
            )
        else:
            raise ValueError(f"unknown Q8 axis {self.axis!r}")
        score = phi * float(result.value) + (1.0 - phi) * wrong
        return score, float(result.value), float(result.bait_share)

    def predict_v(self, weights: NDArray[np.float64]) -> float:
        return self.query(weights)[0]


def fidelity_candidates(spec: PilotSpec) -> list[NDArray[np.float64]]:
    rng = np.random.default_rng(spec.seed0 + 83)
    candidates = [DEFAULT_WEIGHTS.copy(), ENGAGEMENT_WEIGHTS.copy()]
    candidates.extend(from_theta(theta) for theta in sample_theta(rng, spec.m1_sim_draws))
    centre = to_theta(DEFAULT_WEIGHTS)
    local = np.clip(centre + 0.18 * rng.normal(size=(64, centre.size)), LOG_LO, LOG_HI)
    candidates.extend(from_theta(theta) for theta in local)
    return candidates


def m1_against_sim(harness: Harness, spec: PilotSpec, sim: FidelitySim) -> Commit:
    """Search the dialled sim for free, confirm the winner on TrueWorld.

    TrueWorld budget is two labels: default and the sim pick. That is the
    simulate-then-validate pattern, not B5's eleven-measurement screen.
    """
    scored = [(sim.predict_v(weights), weights) for weights in fidelity_candidates(spec)]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_w = scored[0][1].copy()
    shipped = DEFAULT_WEIGHTS.copy()
    best_v = -np.inf
    to_label = [DEFAULT_WEIGHTS]
    if not np.allclose(best_w, DEFAULT_WEIGHTS):
        to_label.append(best_w)
    for weights in to_label:
        try:
            rec = harness.label(weights, spec.m1_label_users)
        except OverBudgetError:
            break
        if rec.unsafe or rec.value is None:
            continue
        if rec.value > best_v:
            best_v = rec.value
            shipped = np.asarray(rec.weights, dtype=np.float64).copy()
    return Commit(
        "M1",
        shipped,
        harness.budget.spent,
        len(harness.history),
        getattr(sim, "tag", f"fidelity_phi_{sim.phi:g}"),
    )
