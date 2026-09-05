"""Search-space packing and pilot budget. Q15 pinned here for phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import DEFAULT_WEIGHTS, N_ACTIONS, POSITIVE_MASK

# Log-magnitudes. Signs are fixed: four positives, two negatives.
LOG_LO = np.log(np.array([0.05, 0.4, 0.01, 0.2, 8.0, 30.0], dtype=np.float64))
LOG_HI = np.log(np.array([10.0, 20.0, 6.0, 10.0, 100.0, 400.0], dtype=np.float64))
SIGNS = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class PilotSpec:
    """Not confirmatory. Stage-2 will replace these after variance is known."""

    budget_user_days: int = 24_000
    search_users: int = 800
    search_days: int = 2
    n_seeds: int = 5
    seed0: int = 10
    eval_rollout_seed: int = 7
    u1_random: int = 8
    u1_es_generations: int = 10
    u1_es_population: int = 12
    u1_es_sigma0: float = 0.5
    u1_es_decay: float = 0.85
    m1_label_users: int = 100
    m1_confirm_users: int = 340
    m1_sim_draws: int = 256
    fidelity_sim_users: int = 400


def to_theta(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(np.clip(np.abs(weights), 1e-9, None))


def from_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(theta, dtype=np.float64), LOG_LO, LOG_HI)
    return SIGNS * np.exp(clipped)


def sample_theta(rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
    u = rng.random((n, N_ACTIONS))
    return LOG_LO + u * (LOG_HI - LOG_LO)


def delayed_label_design(
    spec: PilotSpec, horizon_days: int
) -> list[NDArray[np.float64]]:
    """Fixed configs shared by B5 and M1 under the delayed-label budget."""
    from rl_opt.config import ENGAGEMENT_WEIGHTS

    cost = spec.m1_label_users * horizon_days
    n = spec.budget_user_days // cost
    rng = np.random.default_rng(spec.seed0 + 57)
    weights = [DEFAULT_WEIGHTS.copy(), ENGAGEMENT_WEIGHTS.copy()]
    weights.extend(from_theta(theta) for theta in sample_theta(rng, max(0, n - 2)))
    return weights


def default_theta() -> NDArray[np.float64]:
    return to_theta(DEFAULT_WEIGHTS)


assert POSITIVE_MASK[:4].all() and not POSITIVE_MASK[4:].any()
