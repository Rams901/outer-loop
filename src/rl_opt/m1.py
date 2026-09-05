"""M1 variants for the one allowed, development-seed-only retune."""

from __future__ import annotations

import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS
from rl_opt.harness import Harness, OverBudgetError
from rl_opt.search import Commit
from rl_opt.simworld import GaussianProcessSimWorld, SimWorld
from rl_opt.space import (
    LOG_HI,
    LOG_LO,
    PilotSpec,
    delayed_label_design,
    from_theta,
    sample_theta,
    to_theta,
)


def _collect_labels(harness: Harness, spec: PilotSpec) -> list:
    for weights in delayed_label_design(spec, harness.params.horizon_days):
        try:
            harness.label(weights, spec.m1_label_users)
        except OverBudgetError:
            break
    return harness.labeled()


def _candidate_pool(labeled: list, spec: PilotSpec) -> list[np.ndarray]:
    rng = np.random.default_rng(spec.seed0 + 83)
    candidates = [np.asarray(row.weights, dtype=np.float64) for row in labeled]
    candidates.extend(from_theta(theta) for theta in sample_theta(rng, spec.m1_sim_draws))
    best_rows = sorted(
        (row for row in labeled if row.value is not None),
        key=lambda row: float(row.value),
        reverse=True,
    )[:3]
    for row in best_rows:
        centre = to_theta(np.asarray(row.weights))
        local = np.clip(
            centre + 0.18 * rng.normal(size=(128, centre.size)), LOG_LO, LOG_HI
        )
        candidates.extend(from_theta(theta) for theta in local)
    return candidates


def m1_ridge_no_confirm(harness: Harness, spec: PilotSpec) -> Commit:
    """Use all 11 shared labels, then extrapolate with the original ridge."""
    labeled = _collect_labels(harness, spec)
    if len(labeled) < 3:
        return Commit("M1", DEFAULT_WEIGHTS.copy(), harness.budget.spent, len(harness.history), "fallback_default")
    sim = SimWorld.fit(labeled)
    scored = [(sim.predict_v(w), w) for w in _candidate_pool(labeled, spec)]
    scored.sort(key=lambda t: t[0], reverse=True)
    return Commit(
        "M1",
        scored[0][1].copy(),
        harness.budget.spent,
        len(harness.history),
        "ridge_11_labels_no_confirm",
    )


def _m1_gp(harness: Harness, spec: PilotSpec, beta: float) -> Commit:
    labeled = _collect_labels(harness, spec)
    if len(labeled) < 3:
        return Commit("M1", DEFAULT_WEIGHTS.copy(), harness.budget.spent, len(harness.history), "fallback_default")
    sim = GaussianProcessSimWorld.fit(labeled)
    scored = []
    for weights in _candidate_pool(labeled, spec):
        mean, std = sim.predict(weights)
        scored.append((mean - beta * std, weights))
    scored.sort(key=lambda item: item[0], reverse=True)
    return Commit(
        "M1",
        scored[0][1].copy(),
        harness.budget.spent,
        len(harness.history),
        f"gp_11_labels_lcb_{beta:g}",
    )


def m1_gp_mean(harness: Harness, spec: PilotSpec) -> Commit:
    return _m1_gp(harness, spec, beta=0.0)


def m1_gp_lcb_half(harness: Harness, spec: PilotSpec) -> Commit:
    return _m1_gp(harness, spec, beta=0.5)


def m1_gp_lcb_one(harness: Harness, spec: PilotSpec) -> Commit:
    return _m1_gp(harness, spec, beta=1.0)


# Frozen after `run_m1_dev` selected it on development seeds 20-24.
m1_sim_filter = m1_gp_lcb_one
