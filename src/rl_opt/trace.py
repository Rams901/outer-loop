"""Canonical-scenario tracer. Renderers consume this; they do not recompute scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import (
    ACTIONS,
    DEFAULT_WEIGHTS,
    ENGAGEMENT_WEIGHTS,
    N_ACTIONS,
)
from rl_opt.ranker import score_candidates
from rl_opt.world import (
    SAT_INIT,
    TrueWorld,
    item_utility,
    sample_candidates,
    satisfaction_delta,
)

SCHEMA = "rl_opt.trace.v1"


def _probs_dict(row: NDArray[np.float64]) -> dict[str, float]:
    return {name: float(row[i]) for i, name in enumerate(ACTIONS)}


def _terms_dict(pred: NDArray[np.float64], weights: NDArray[np.float64]) -> dict[str, float]:
    parts = pred * weights
    out = {name: float(parts[i]) for i, name in enumerate(ACTIONS)}
    out["total"] = float(parts.sum())
    return out


@dataclass(frozen=True)
class CanonicalPick:
    user_id: int
    candidate_ids: tuple[int, ...]
    scenario_seed: int


def pick_canonical_user(world: TrueWorld, scenario_seed: int = 11) -> CanonicalPick:
    """User whose day-0 candidates include bait and quality, and whose two configs disagree."""
    rng = np.random.default_rng(scenario_seed)
    p = world.params
    cand = sample_candidates(rng, p.n_users, p.n_posts, p.n_candidates)
    pred = world.predicted_probs()
    is_bait = world.catalogue.bait > 0.5
    is_quality = (world.catalogue.quality > 0.55) & ~is_bait

    best: CanonicalPick | None = None
    best_key = (-1.0, 1.0)
    for u in range(p.n_users):
        ids = cand[u]
        if not (is_bait[ids].any() and is_quality[ids].any()):
            continue
        s_def = score_candidates(pred[u, ids], DEFAULT_WEIGHTS)
        s_eng = score_candidates(pred[u, ids], ENGAGEMENT_WEIGHTS)
        order_d = np.argsort(-s_def)
        order_e = np.argsort(-s_eng)
        if ids[order_d[0]] == ids[order_e[0]]:
            continue
        k = p.feed_size
        top_d = ids[order_d[:k]]
        top_e = ids[order_e[:k]]
        bait_d = float(is_bait[top_d].mean())
        bait_e = float(is_bait[top_e].mean())
        if bait_e <= bait_d:
            continue
        overlap = len(set(top_d.tolist()) & set(top_e.tolist())) / k
        key = (bait_e - bait_d, -overlap)
        if key > best_key:
            best_key = key
            best = CanonicalPick(user_id=u, candidate_ids=tuple(int(i) for i in ids), scenario_seed=scenario_seed)
    if best is None:
        raise RuntimeError("no user with bait/quality candidates and rank disagreement")
    return best


def trace_scenario(
    world: TrueWorld,
    user_id: int,
    candidate_ids: NDArray[np.int64] | tuple[int, ...],
    configs: dict[str, NDArray[np.float64]] | None = None,
    day: int = 0,
    satisfaction_before: float = SAT_INIT,
) -> dict[str, Any]:
    configs = configs or {"default": DEFAULT_WEIGHTS, "engagement": ENGAGEMENT_WEIGHTS}
    ids = np.asarray(candidate_ids, dtype=np.int64)
    cat = world.catalogue
    pred = world.predicted_probs()[user_id, ids]
    true = world.true_probs()[user_id, ids]
    rel = world.relevance()[user_id, ids]
    k = world.params.feed_size
    pop = world.population

    per_config: dict[str, dict[str, Any]] = {}
    ranks: dict[str, NDArray[np.int64]] = {}
    for name, w in configs.items():
        scores = score_candidates(pred, w)
        order = np.argsort(-scores)
        ranks[name] = order
        shown = ids[order[:k]]
        shown_local = order[:k]
        q, b, t, r = cat.quality[shown], cat.bait[shown], cat.toxicity[shown], rel[shown_local]
        util = item_utility(q, b, t, r)
        delta = float(satisfaction_delta(float(pop.sensitivity[user_id]), float(util.mean())))
        per_config[name] = {
            "weights": [float(x) for x in w],
            "slots": [int(x) for x in shown],
            "mean_bait": float((b > 0.5).mean()),
            "delta_s": delta,
            "satisfaction_after": float(np.clip(satisfaction_before + delta, 0.02, 0.98)),
        }

    candidates = []
    for i, post_id in enumerate(ids):
        entry: dict[str, Any] = {
            "post_id": int(post_id),
            "quality": float(cat.quality[post_id]),
            "bait": float(cat.bait[post_id]),
            "toxicity": float(cat.toxicity[post_id]),
            "cold_start": bool(cat.cold_start[post_id]),
            "relevance": float(rel[i]),
            "true_p": _probs_dict(true[i]),
            "pred_p": _probs_dict(pred[i]),
            "score_terms": {name: _terms_dict(pred[i], w) for name, w in configs.items()},
            "rank": {},
            "shown": {},
        }
        for name, order in ranks.items():
            rank = int(np.where(order == i)[0][0]) + 1
            entry["rank"][name] = rank
            entry["shown"][name] = rank <= k
        candidates.append(entry)

    return {
        "schema": SCHEMA,
        "world_seed": world.seed,
        "day": day,
        "user_id": user_id,
        "user": {
            "interests": [float(x) for x in pop.interests[user_id]],
            "sensitivity": float(pop.sensitivity[user_id]),
            "satisfaction_before": satisfaction_before,
        },
        "configs": {name: {"weights": [float(x) for x in w]} for name, w in configs.items()},
        "candidates": candidates,
        "feed": per_config,
    }


def terms_sum_ok(entry: dict[str, Any], atol: float = 1e-9) -> bool:
    for terms in entry["score_terms"].values():
        total = terms["total"]
        parts = sum(terms[a] for a in ACTIONS)
        if abs(total - parts) > atol:
            return False
    return True


def as_jsonable(trace: dict[str, Any]) -> dict[str, Any]:
    return asdict(trace) if not isinstance(trace, dict) else trace
