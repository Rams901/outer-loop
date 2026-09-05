"""Runner for the step-0 diagnostic.

Answers one question: after a genuine search with out-of-sample confirmation,
how much headroom is there above the default? That number is the denominator of
the primary metric, so everything downstream depends on it.

Runs seeds sequentially on purpose. Parallel numpy workers oversubscribe BLAS
and take longer on this machine.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rl_opt.config import WorldParams
from rl_opt.space import PilotSpec
from rl_opt.u1_strong import StrongSpec, u1_strong
from rl_opt.world import TrueWorld


def _one(world_seed: int) -> dict:
    world = TrueWorld.generate(WorldParams(), seed=world_seed)
    res = u1_strong(world, world_seed, StrongSpec())
    return asdict(res)


def main() -> int:
    spec = PilotSpec()
    seeds = [spec.seed0 + i for i in range(spec.n_seeds)]
    rows = []
    for s in seeds:
        print(f"  seed {s} …", flush=True)
        row = _one(s)
        rows.append(row)
        print(
            f"    V0={row['v0_confirm']:.4f}  V*={row['v_star_confirm']:.4f}  "
            f"headroom={row['headroom']:+.4f} ± {row['headroom_se']:.4f}  "
            f"planted={row['w_star_is_planted']}  calls={row['n_calls']}",
            flush=True,
        )

    weak = _load_weak()
    heads = np.array([r["headroom"] for r in rows])
    summary = {
        "phase": "0-diagnostic",
        "question": "Is V* - V0 real, or best-of-30 selection bias at a single rollout seed?",
        "protocol": {
            "search_seeds": list(StrongSpec().search_seeds),
            "confirm_seeds": list(StrongSpec().confirm_seeds),
            "note": "search and confirm rollout seeds are disjoint; V0 uses the identical confirm protocol",
        },
        "runs": rows,
        "mean_headroom_strong": float(heads.mean()),
        "mean_headroom_weak": weak["mean"] if weak else None,
        "per_seed": [
            {
                "world_seed": r["world_seed"],
                "v0": r["v0_confirm"],
                "v_star": r["v_star_confirm"],
                "headroom_strong": r["headroom"],
                "headroom_se": r["headroom_se"],
                "headroom_weak": (weak["per_seed"].get(str(r["world_seed"])) if weak else None),
                "w_star_is_planted": r["w_star_is_planted"],
                "n_calls": r["n_calls"],
            }
            for r in rows
        ],
    }
    out = Path(__file__).resolve().parents[2] / "analysis" / "phase0_u1_strong.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print("step 0 - strengthened oracle")
    print(f"  {rows[0]['n_calls']} oracle calls/seed, {rows[0]['n_candidates']} candidates")
    print(f"  {'seed':>5} {'V0':>8} {'V*':>8} {'headroom':>10} {'+/-':>8} {'weak U1':>9}  planted")
    for r in summary["per_seed"]:
        wk = f"{r['headroom_weak']:.4f}" if r["headroom_weak"] is not None else "   n/a"
        print(
            f"  {r['world_seed']:>5} {r['v0']:>8.4f} {r['v_star']:>8.4f} "
            f"{r['headroom_strong']:>10.4f} {r['headroom_se']:>8.4f} {wk:>9}  {r['w_star_is_planted']}"
        )
    print(f"  mean headroom  strong={heads.mean():.4f}   weak={weak['mean']:.4f}" if weak else "")
    print(f"  wrote {out}")
    return 0


def _load_weak() -> dict | None:
    p = Path(__file__).resolve().parents[2] / "analysis" / "phase1_pilot.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    per = {str(r["world_seed"]): r["v_star"] - r["v0"] for r in data["runs"]}
    return {"per_seed": per, "mean": float(np.mean(list(per.values())))}


if __name__ == "__main__":
    raise SystemExit(main())
