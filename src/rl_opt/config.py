"""Policy vector and world hyperparameters.

Six actions, matching the Q11 recommendation: two common positives, one rarer
effortful positive, one continuous-flavoured positive (dwell), one mid-rare
negative, one very rare negative. Magnitudes echo `param.rs` so inverse-propensity
scaling is in the problem, not because we claim those numbers transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS: tuple[str, ...] = ("like", "reply", "dwell", "share", "hide", "report")
N_ACTIONS = len(ACTIONS)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}

# Production compromise: inverse-propensity magnitudes (report ≪ like) with a
# mild cheap-engagement tilt. Hide/report match param.rs; like/dwell are higher
# and share/reply lower than a pure quality ranking, so default is beatable.
DEFAULT_WEIGHTS = np.array([1.8, 3.5, 0.55, 1.0, -43.2, -234.0], dtype=np.float64)

# Hand-built trap: further along the same axis. Feed mix shifts, does not collapse.
ENGAGEMENT_WEIGHTS = np.array([2.6, 3.0, 1.6, 1.2, -32.0, -90.0], dtype=np.float64)

POSITIVE_MASK = DEFAULT_WEIGHTS > 0
NEGATIVE_MASK = DEFAULT_WEIGHTS < 0


@dataclass(frozen=True)
class WorldParams:
    n_users: int = 4000
    n_posts: int = 240
    n_topics: int = 4
    n_candidates: int = 48
    feed_size: int = 12
    horizon_days: int = 21
    proxy_days: int = 2
    bait_fraction: float = 0.22
    cold_start_fraction: float = 0.15
    position_decay: float = 0.72
    pred_noise: float = 0.35
    cold_start_noise: float = 1.1
    propensity_spread: float = 0.45
    interest_concentration: float = 0.7
    bait_detectability: float = 1.0
    supply_feedback: bool = False
    report_floor: float = 2.5e-4
    hide_floor: float = 8.0e-3

    def __post_init__(self) -> None:
        if self.supply_feedback:
            raise ValueError("supply_feedback is an ablation and is off in phase 0")
        if self.feed_size > self.n_candidates:
            raise ValueError("feed_size cannot exceed n_candidates")
        if self.proxy_days >= self.horizon_days:
            raise ValueError("proxy_days must be strictly shorter than horizon_days")
