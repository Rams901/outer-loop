"""Entrants B0–B5.

B0–B4 maximise the observable 2-day proxy, then commit. B5 is different on
purpose: it spends the identical budget on full-horizon V labels and ships the
best config it actually measured. B5 exists so that M1's surrogate is compared
against a method with the *same* information channel rather than against
baselines that were forbidden from seeing V.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import DEFAULT_WEIGHTS
from rl_opt.harness import Harness, OverBudgetError, Receipt
from rl_opt.space import (
    LOG_HI,
    LOG_LO,
    PilotSpec,
    delayed_label_design,
    from_theta,
    sample_theta,
    to_theta,
)


@dataclass
class Commit:
    name: str
    weights: NDArray[np.float64]
    spent: int
    n_experiments: int
    chosen_by: str


def _run_until_broke(harness: Harness, spec: PilotSpec, weights: NDArray[np.float64]) -> Receipt | None:
    try:
        return harness.experiment(weights, spec.search_users, spec.search_days)
    except OverBudgetError:
        return None


def b0_default(harness: Harness, spec: PilotSpec) -> Commit:
    return Commit("B0", DEFAULT_WEIGHTS.copy(), harness.budget.spent, 0, "ship_default")


def b1_grid(harness: Harness, spec: PilotSpec) -> Commit:
    """One-axis A/B on like-weight, four values. Incumbent."""
    likes = [1.8, 2.4, 3.2, 5.0]
    best_w = DEFAULT_WEIGHTS.copy()
    best_proxy = -np.inf
    for like in likes:
        w = DEFAULT_WEIGHTS.copy()
        w[0] = like
        rec = _run_until_broke(harness, spec, w)
        if rec is None:
            break
        if rec.unsafe:
            continue
        if rec.proxy > best_proxy:
            best_proxy = rec.proxy
            best_w = w
    return Commit("B1", best_w, harness.budget.spent, len(harness.history), "max_proxy_grid")


def b2_random(harness: Harness, spec: PilotSpec) -> Commit:
    rng = np.random.default_rng(spec.seed0 + 91)
    best_w = DEFAULT_WEIGHTS.copy()
    best_proxy = -np.inf
    while True:
        w = from_theta(sample_theta(rng)[0])
        rec = _run_until_broke(harness, spec, w)
        if rec is None:
            break
        if rec.unsafe:
            continue
        if rec.proxy > best_proxy:
            best_proxy = rec.proxy
            best_w = w
    return Commit("B2", best_w, harness.budget.spent, len(harness.history), "max_proxy_random")


def _fit_gp(X: NDArray[np.float64], y: NDArray[np.float64], ls: float = 0.8, noise: float = 0.02):
    n = len(y)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    k = np.exp(-0.5 * d2 / (ls**2)) + noise * np.eye(n)
    jitter = 1e-6
    for _ in range(5):
        try:
            L = np.linalg.cholesky(k)
            break
        except np.linalg.LinAlgError:
            k = k + jitter * np.eye(n)
            jitter *= 10
    else:
        L = np.linalg.cholesky(k + np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - y.mean()))
    return L, alpha, float(y.mean()), ls, noise


def _gp_ei(theta: NDArray[np.float64], X: NDArray[np.float64], L, alpha, ymean, ybest, ls, noise):
    d2 = ((X - theta) ** 2).sum(axis=1)
    k_star = np.exp(-0.5 * d2 / (ls**2))
    v = np.linalg.solve(L, k_star)
    mu = ymean + k_star @ alpha
    var = max(1e-8, 1.0 + noise - v @ v)
    std = np.sqrt(var)
    z = (mu - ybest) / std
    # erf-based Φ, φ
    from math import erf, exp, sqrt, pi

    cdf = 0.5 * (1.0 + erf(z / sqrt(2)))
    pdf = exp(-0.5 * z * z) / sqrt(2 * pi)
    return (mu - ybest) * cdf + std * pdf


def b3_bo(harness: Harness, spec: PilotSpec) -> Commit:
    rng = np.random.default_rng(spec.seed0 + 17)
    X: list[NDArray[np.float64]] = []
    y: list[float] = []
    best_w = DEFAULT_WEIGHTS.copy()
    best_proxy = -np.inf
    # seed with default + a few random
    starters = [to_theta(DEFAULT_WEIGHTS)] + [sample_theta(rng)[0] for _ in range(3)]
    for theta in starters:
        rec = _run_until_broke(harness, spec, from_theta(theta))
        if rec is None:
            break
        if rec.unsafe:
            continue
        X.append(theta)
        y.append(rec.proxy)
        if rec.proxy > best_proxy:
            best_proxy = rec.proxy
            best_w = from_theta(theta)
    if not X:
        return Commit("B3", best_w, harness.budget.spent, len(harness.history), "max_proxy_bo")
    while True:
        rec = None
        L, alpha, ymean, ls, noise = _fit_gp(np.vstack(X), np.array(y))
        ybest = float(np.max(y))
        cand = sample_theta(rng, 64)
        ei = [_gp_ei(t, np.vstack(X), L, alpha, ymean, ybest, ls, noise) for t in cand]
        theta = cand[int(np.argmax(ei))]
        rec = _run_until_broke(harness, spec, from_theta(theta))
        if rec is None:
            break
        if rec.unsafe:
            continue
        X.append(theta)
        y.append(rec.proxy)
        if rec.proxy > best_proxy:
            best_proxy = rec.proxy
            best_w = from_theta(theta)
    return Commit("B3", best_w, harness.budget.spent, len(harness.history), "max_proxy_bo")


def b4_cma(harness: Harness, spec: PilotSpec) -> Commit:
    """(μ/μ,λ)-style step on log-weights. Not a full CMA-ES; enough as a different search."""
    rng = np.random.default_rng(spec.seed0 + 23)
    mean = to_theta(DEFAULT_WEIGHTS)
    sigma = 0.35
    best_w = DEFAULT_WEIGHTS.copy()
    best_proxy = -np.inf
    while True:
        lam = 4
        thetas = mean + sigma * rng.normal(size=(lam, mean.size))
        thetas = np.clip(thetas, LOG_LO, LOG_HI)
        scored: list[tuple[float, NDArray[np.float64]]] = []
        stopped = False
        for theta in thetas:
            rec = _run_until_broke(harness, spec, from_theta(theta))
            if rec is None:
                stopped = True
                break
            if rec.unsafe:
                continue
            scored.append((rec.proxy, theta))
            if rec.proxy > best_proxy:
                best_proxy = rec.proxy
                best_w = from_theta(theta)
        if stopped or not scored:
            break
        scored.sort(key=lambda t: t[0], reverse=True)
        parents = np.stack([t[1] for t in scored[: max(1, len(scored) // 2)]])
        mean = parents.mean(axis=0)
        sigma = max(0.08, 0.9 * sigma)
    return Commit("B4", best_w, harness.budget.spent, len(harness.history), "max_proxy_es")


def b5_labels(harness: Harness, spec: PilotSpec) -> Commit:
    """Buy full-horizon V labels until broke, ship the best one measured.

    No surrogate and no extrapolation: B5 can only choose among configs it
    actually paid to observe. That is the whole contrast with M1, which pays
    for fewer labels and then queries a fitted model for free.

    Uses the exact label design and observation RNG M1 receives. This isolates
    the fitted model: B5 selects the best measured row; M1 may extrapolate.
    `QUALITY_WEIGHTS` remains withheld because it is oracle-derived.
    """
    best_w = DEFAULT_WEIGHTS.copy()
    best_v = -np.inf
    for w in delayed_label_design(spec, harness.params.horizon_days):
        try:
            rec = harness.label(w, spec.m1_label_users)
        except OverBudgetError:
            break
        if rec.unsafe or rec.value is None:
            continue
        if rec.value > best_v:
            best_v = rec.value
            best_w = w.copy()
    return Commit("B5", best_w, harness.budget.spent, len(harness.history), "max_measured_v")
