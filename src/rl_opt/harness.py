"""Budgeted access to TrueWorld.

Short experiments hide V so search cannot Goodhart the objective.
`label()` charges a full-horizon slice and writes delayed V onto the receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import WorldParams
from rl_opt.metrics import RolloutResult
from rl_opt.world import TrueWorld


class OverBudgetError(RuntimeError):
    pass


@dataclass
class Budget:
    remaining: int
    spent: int = 0

    def charge(self, user_days: int) -> None:
        if user_days < 0:
            raise ValueError("user_days must be >= 0")
        if user_days > self.remaining:
            raise OverBudgetError(f"need {user_days}, have {self.remaining}")
        self.remaining -= user_days
        self.spent += user_days


@dataclass(frozen=True)
class Receipt:
    """One charged experiment. `value` is set only when n_days >= horizon."""

    proxy: float
    unsafe: bool
    user_days: int
    n_users: int
    n_days: int
    hide_rate: float
    report_rate: float
    bait_share: float
    value: float | None
    weights: tuple[float, ...]


@dataclass
class Harness:
    world: TrueWorld
    budget: Budget
    rng: np.random.Generator
    _n_experiments: int = 0
    history: list[Receipt] = field(default_factory=list)

    @property
    def params(self) -> WorldParams:
        return self.world.params

    def experiment(self, weights: NDArray[np.float64], n_users: int, n_days: int) -> Receipt:
        n_users = min(int(n_users), self.params.n_users)
        n_days = int(n_days)
        cost = n_users * n_days
        self.budget.charge(cost)
        ids = self.rng.choice(self.params.n_users, size=n_users, replace=False)
        result = self.world.rollout(weights, n_days=n_days, seed=self._n_experiments, user_ids=ids)
        self._n_experiments += 1
        reveal_v = n_days >= self.params.horizon_days
        rec = Receipt(
            proxy=result.proxy,
            unsafe=result.unsafe,
            user_days=cost,
            n_users=n_users,
            n_days=n_days,
            hide_rate=result.hide_rate,
            report_rate=result.report_rate,
            bait_share=result.bait_share,
            value=result.value if reveal_v else None,
            weights=tuple(float(x) for x in np.asarray(weights, dtype=np.float64)),
        )
        self.history.append(rec)
        return rec

    def labeled(self) -> list[Receipt]:
        """Rows where delayed V was paid for (full horizon)."""
        return [r for r in self.history if r.value is not None]

    def label(self, weights: NDArray[np.float64], n_users: int) -> Receipt:
        """Charge a full-horizon slice. This is how delayed return enters the log."""
        return self.experiment(weights, n_users, n_days=self.params.horizon_days)


@dataclass
class Oracle:
    """Unlimited V evaluations. Not an entrant. Computes w* and scores commits."""

    world: TrueWorld
    rollout_seed: int = 7
    calls: int = 0

    def value(self, weights: NDArray[np.float64]) -> RolloutResult:
        self.calls += 1
        return self.world.rollout(weights, seed=self.rollout_seed)
