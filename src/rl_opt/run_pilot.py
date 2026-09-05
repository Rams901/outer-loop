"""Phase 1 pilot: B0–B4 under equal budget, regret vs U1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_opt.config import WorldParams
from rl_opt.harness import Budget, Harness, Oracle
from rl_opt.oracle_search import u1_oracle
from rl_opt.search import b0_default, b1_grid, b2_random, b3_bo, b4_cma
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld

UNSAFE_R = 2.0
METHODS = (
    ("B0", b0_default),
    ("B1", b1_grid),
    ("B2", b2_random),
    ("B3", b3_bo),
    ("B4", b4_cma),
)
METHOD_SEED = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4}


def normalised_regret(v_hat: float, v_star: float, v0: float, unsafe: bool) -> float:
    if unsafe:
        return UNSAFE_R
    denom = v_star - v0
    if abs(denom) < 1e-6:
        # U1 did not beat the default; the registered ratio is undefined.
        return float((v_star - v_hat) / max(abs(v_star), 1e-6))
    return float((v_star - v_hat) / denom)


def run_seed(world_seed: int, spec: PilotSpec, params: WorldParams) -> dict:
    world = TrueWorld.generate(params, seed=world_seed)
    oracle = Oracle(world, rollout_seed=spec.eval_rollout_seed)
    w_star, v_star, v0 = u1_oracle(oracle, spec)
    rows = {}
    for name, fn in METHODS:
        harness = Harness(
            world,
            Budget(spec.budget_user_days),
            np.random.default_rng(world_seed * 1009 + METHOD_SEED[name]),
        )
        commit = fn(harness, spec)
        scored = oracle.value(commit.weights)
        rows[name] = {
            "spent": commit.spent,
            "n_experiments": commit.n_experiments,
            "proxy": scored.proxy,
            "value": scored.value,
            "unsafe": scored.unsafe,
            "bait_share": scored.bait_share,
            "regret": normalised_regret(scored.value, v_star, v0, scored.unsafe),
            "weights": commit.weights.tolist(),
        }
    default_proxy = rows["B0"]["proxy"]
    for name in rows:
        rows[name]["walkback"] = bool(
            rows[name]["proxy"] > default_proxy + 1e-6 and rows[name]["value"] < v0 - 1e-4
        )
    return {
        "world_seed": world_seed,
        "v_star": v_star,
        "v0": v0,
        "w_star": w_star.tolist(),
        "u1_calls": oracle.calls,
        "methods": rows,
    }


def main() -> int:
    spec = PilotSpec()
    params = WorldParams()
    seeds = [spec.seed0 + i for i in range(spec.n_seeds)]
    runs = [run_seed(s, spec, params) for s in seeds]
    regrets = {name: [run["methods"][name]["regret"] for run in runs] for name, _ in METHODS}
    mean_value = {name: float(np.mean([run["methods"][name]["value"] for run in runs])) for name, _ in METHODS}
    summary = {
        "phase": 1,
        "spec": {
            "budget_user_days": spec.budget_user_days,
            "search_users": spec.search_users,
            "search_days": spec.search_days,
            "n_seeds": spec.n_seeds,
        },
        "runs": runs,
        "mean_regret": {k: float(np.mean(v)) for k, v in regrets.items()},
        "mean_value": mean_value,
        "c2_random_not_trivial": all(r >= 0.2 for r in regrets["B2"]),
        "c3_b3_beats_b1": float(np.mean(regrets["B3"])) < float(np.mean(regrets["B1"])) - 1e-6,
        "p1_exit_b4_beats_b1": float(np.mean(regrets["B4"])) < float(np.mean(regrets["B1"])) - 1e-6,
    }
    out = Path(__file__).resolve().parents[2] / "analysis" / "phase1_pilot.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        from rl_opt.figures import plot_pilot_bars

        plot_pilot_bars(summary, out.parent / "figures" / "p1_committed_v.png")
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    print("phase 1 pilot")
    print(f"  budget {spec.budget_user_days} user-days  ·  {spec.n_seeds} seeds")
    for name, _ in METHODS:
        rs = regrets[name]
        print(f"  {name}  mean R={np.mean(rs):.3f}  [{', '.join(f'{x:.2f}' for x in rs)}]")
    print(f"  C2 (B2 never R<0.2): {summary['c2_random_not_trivial']}")
    print(f"  C3 (mean R_B3 < R_B1): {summary['c3_b3_beats_b1']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
