"""C4: the best per-segment config beats the best global config by ≥ 20% of headroom.

Segments are primary topic (argmax of user interests). This gates the contextual
extension; it does not enter confirmatory H1. Oracle only — not an entrant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.oracle_search import QUALITY_WEIGHTS, _axis_neighbours
from rl_opt.space import PilotSpec, from_theta, sample_theta
from rl_opt.world import TrueWorld

C4_THRESHOLD = 0.20
C4_SEEDS = (0, 1, 2)
MIN_SEGMENT = 80


def _pool(spec: PilotSpec) -> list[NDArray[np.float64]]:
    rng = np.random.default_rng(spec.seed0 + 11)
    pool = [DEFAULT_WEIGHTS.copy(), ENGAGEMENT_WEIGHTS.copy(), QUALITY_WEIGHTS.copy()]
    pool.extend(_axis_neighbours())
    pool.extend(from_theta(theta) for theta in sample_theta(rng, 6))
    return pool


def _value_on(
    world: TrueWorld, weights: NDArray[np.float64], user_ids: NDArray[np.int64] | None, seed: int
) -> float:
    return float(world.rollout(weights, seed=seed, user_ids=user_ids).value)


def evaluate_seed(world_seed: int, spec: PilotSpec, params: WorldParams) -> dict:
    world = TrueWorld.generate(params, seed=world_seed)
    pool = _pool(spec)
    eval_seed = spec.eval_rollout_seed
    v_global = [_value_on(world, w, None, eval_seed) for w in pool]
    i_star = int(np.argmax(v_global))
    v_star = float(v_global[i_star])
    v0 = _value_on(world, DEFAULT_WEIGHTS, None, eval_seed)
    headroom = v_star - v0

    topics = world.primary_topic()
    n_users = params.n_users
    segments = []
    combined = 0.0
    for topic in range(params.n_topics):
        ids = np.flatnonzero(topics == topic).astype(np.int64)
        if ids.size < MIN_SEGMENT:
            continue
        values = [_value_on(world, w, ids, eval_seed) for w in pool]
        best_i = int(np.argmax(values))
        v_s = float(values[best_i])
        weight = ids.size / n_users
        combined += weight * v_s
        segments.append(
            {
                "topic": topic,
                "n_users": int(ids.size),
                "v_segment": v_s,
                "v_global_star_on_segment": _value_on(world, pool[i_star], ids, eval_seed),
                "weights": pool[best_i].tolist(),
            }
        )
    covered = sum(s["n_users"] for s in segments)
    if covered < n_users:
        leftover = n_users - covered
        combined += (leftover / n_users) * v_star
    gap = combined - v_star
    ratio = float(gap / headroom) if abs(headroom) > 1e-6 else 0.0
    return {
        "world_seed": world_seed,
        "v0": v0,
        "v_star_global": v_star,
        "v_star_segmented": float(combined),
        "headroom": float(headroom),
        "gap": float(gap),
        "gap_over_headroom": ratio,
        "passed": bool(ratio >= C4_THRESHOLD),
        "segments": segments,
    }


def main() -> int:
    spec = PilotSpec()
    params = WorldParams()
    rows = [evaluate_seed(s, spec, params) for s in C4_SEEDS]
    mean_ratio = float(np.mean([r["gap_over_headroom"] for r in rows]))
    passed = all(r["passed"] for r in rows)
    report = {
        "criterion": "C4",
        "threshold": C4_THRESHOLD,
        "seeds": list(C4_SEEDS),
        "runs": rows,
        "mean_gap_over_headroom": mean_ratio,
        "passed": passed,
        "note": (
            "Oracle pool per primary-topic segment. Gates the contextual extension; "
            "not part of confirmatory H1."
        ),
    }
    out = Path(__file__).resolve().parents[2] / "analysis" / "phase0_c4.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("C4  (heterogeneity / contextual gate)")
    print(f"  threshold: gap / (V*−V0) ≥ {C4_THRESHOLD:.0%}")
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(
            f"  seed {r['world_seed']}: V*={r['v_star_global']:.3f}  "
            f"V_seg={r['v_star_segmented']:.3f}  gap/headroom={r['gap_over_headroom']:.3f}  [{mark}]"
        )
    print(f"  mean ratio={mean_ratio:.3f}  wrote {out}")
    print("  C4 " + ("passed" if passed else "failed"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
