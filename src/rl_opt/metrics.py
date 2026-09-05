"""Aggregate metrics from a rollout. Proxy vs true objective is the whole point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import ACTION_INDEX, WorldParams


@dataclass(frozen=True)
class RolloutResult:
    proxy: float
    value: float
    like_rate: float
    reply_rate: float
    dwell_rate: float
    share_rate: float
    hide_rate: float
    report_rate: float
    bait_share: float
    mean_satisfaction: float
    n_impressions: int
    n_user_days: int
    n_active_user_days: int
    unsafe: bool
    daily_return: tuple[float, ...]
    daily_bait_share: tuple[float, ...]
    daily_engagement: tuple[float, ...]
    daily_satisfaction: tuple[float, ...]

    def proxy_lift(self, baseline: "RolloutResult") -> float:
        if baseline.proxy == 0.0:
            raise ZeroDivisionError("baseline proxy is zero")
        return self.proxy / baseline.proxy - 1.0


def safety_violated(hide_rate: float, report_rate: float, params: WorldParams) -> bool:
    return hide_rate > params.hide_floor or report_rate > params.report_floor


def from_counters(
    impressions: int,
    actions: NDArray[np.int64],
    bait_shown: int,
    returned: NDArray[np.int8],
    satisfaction: NDArray[np.float64],
    proxy_engagement: int,
    proxy_impressions: int,
    params: WorldParams,
    daily_return: NDArray[np.float64] | None = None,
    daily_bait_share: NDArray[np.float64] | None = None,
    daily_engagement: NDArray[np.float64] | None = None,
    daily_satisfaction: NDArray[np.float64] | None = None,
) -> RolloutResult:
    """actions is (n_actions,) counts over the full horizon."""
    if impressions == 0:
        raise ValueError("rollout produced no impressions")
    rates = actions.astype(np.float64) / impressions
    hide = float(rates[ACTION_INDEX["hide"]])
    report = float(rates[ACTION_INDEX["report"]])
    proxy = proxy_engagement / max(proxy_impressions, 1)
    # V: mean daily return from day 1 onward. Day 0 is everyone by construction.
    value = float(returned[1:].mean()) if returned.size > 1 else float(returned.mean())
    return RolloutResult(
        proxy=float(proxy),
        value=value,
        like_rate=float(rates[ACTION_INDEX["like"]]),
        reply_rate=float(rates[ACTION_INDEX["reply"]]),
        dwell_rate=float(rates[ACTION_INDEX["dwell"]]),
        share_rate=float(rates[ACTION_INDEX["share"]]),
        hide_rate=hide,
        report_rate=report,
        bait_share=bait_shown / impressions,
        mean_satisfaction=float(satisfaction.mean()),
        n_impressions=int(impressions),
        n_user_days=int(returned.size),
        n_active_user_days=int(returned.sum()),
        unsafe=safety_violated(hide, report, params),
        daily_return=tuple(float(x) for x in (daily_return if daily_return is not None else [])),
        daily_bait_share=tuple(float(x) for x in (daily_bait_share if daily_bait_share is not None else [])),
        daily_engagement=tuple(float(x) for x in (daily_engagement if daily_engagement is not None else [])),
        daily_satisfaction=tuple(float(x) for x in (daily_satisfaction if daily_satisfaction is not None else [])),
    )
