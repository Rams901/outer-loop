"""Phase 0 exit: exhibit a hand-built config that Goodharts the proxy."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.metrics import RolloutResult
from rl_opt.world import TrueWorld

MIN_PROXY_LIFT = 0.05
PILOT_SEEDS = (0, 1, 2)


def _result_dict(r: RolloutResult) -> dict:
    return {
        "proxy": r.proxy,
        "value": r.value,
        "like_rate": r.like_rate,
        "reply_rate": r.reply_rate,
        "dwell_rate": r.dwell_rate,
        "share_rate": r.share_rate,
        "hide_rate": r.hide_rate,
        "report_rate": r.report_rate,
        "bait_share": r.bait_share,
        "mean_satisfaction": r.mean_satisfaction,
        "n_impressions": r.n_impressions,
        "unsafe": r.unsafe,
    }


def evaluate_seed(params: WorldParams, world_seed: int, rollout_seed: int = 7) -> dict:
    world = TrueWorld.generate(params, seed=world_seed)
    default = world.rollout(DEFAULT_WEIGHTS, seed=rollout_seed)
    bait = world.rollout(ENGAGEMENT_WEIGHTS, seed=rollout_seed)
    lift = bait.proxy_lift(default)
    c1 = lift >= MIN_PROXY_LIFT and bait.value < default.value
    return {
        "world_seed": world_seed,
        "base_rates": world.action_base_rates(),
        "like_rate_bait_posts": world.bait_vs_quality_likes()[0],
        "like_rate_quality_posts": world.bait_vs_quality_likes()[1],
        "default": _result_dict(default),
        "engagement": _result_dict(bait),
        "proxy_lift": lift,
        "value_delta": bait.value - default.value,
        "c1": c1,
    }


def main() -> int:
    params = WorldParams()
    rows = [evaluate_seed(params, s) for s in PILOT_SEEDS]
    mean_lift = float(np.mean([r["proxy_lift"] for r in rows]))
    mean_value_delta = float(np.mean([r["value_delta"] for r in rows]))
    passed = all(r["c1"] for r in rows)
    report = {
        "phase": 0,
        "criterion": "C1",
        "min_proxy_lift": MIN_PROXY_LIFT,
        "params": asdict(params),
        "default_weights": DEFAULT_WEIGHTS.tolist(),
        "engagement_weights": ENGAGEMENT_WEIGHTS.tolist(),
        "seeds": rows,
        "mean_proxy_lift": mean_lift,
        "mean_value_delta": mean_value_delta,
        "passed": passed,
    }
    out_dir = Path(__file__).resolve().parents[2] / "analysis"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "phase0_c1.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    print("phase 0 / C1")
    print(f"  proxy lift mean: {mean_lift:.3%}  (need ≥ {MIN_PROXY_LIFT:.0%})")
    print(f"  V_bait − V_default: {mean_value_delta:+.4f}  (need < 0)")
    for r in rows:
        mark = "PASS" if r["c1"] else "FAIL"
        print(
            f"  seed {r['world_seed']}: lift={r['proxy_lift']:.3%}  "
            f"ΔV={r['value_delta']:+.4f}  bait_share {r['default']['bait_share']:.3f}→"
            f"{r['engagement']['bait_share']:.3f}  [{mark}]"
        )
        rates = r["base_rates"]
        print(
            "    base rates  like={like:.3g}  reply={reply:.3g}  dwell={dwell:.3g}  "
            "share={share:.3g}  hide={hide:.3g}  report={report:.3g}".format(**rates)
        )
    print(f"  wrote {path}")
    print("  C1 " + ("passed" if passed else "failed"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
