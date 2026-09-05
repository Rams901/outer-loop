"""Exploratory M1 vs B0/B1/B3/B4/B5 on the phase-1 seeds. Not confirmatory.

B5 is the comparison that matters. B0-B4 never see V, so M1 beating them only
shows that delayed labels help, which we already knew. B5 spends the identical
budget on delayed labels without a surrogate, so M1 vs B5 isolates the thing
the project is actually about: does fitting a model beat buying more
measurements?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_opt.config import WorldParams
from rl_opt.figures import plot_committed_v
from rl_opt.harness import Budget, Harness, Oracle
from rl_opt.m1 import m1_sim_filter
from rl_opt.oracle_search import u1_oracle
from rl_opt.run_pilot import normalised_regret
from rl_opt.search import b0_default, b1_grid, b3_bo, b4_cma, b5_labels
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld

METHODS = (
    ("B0", b0_default),
    ("B1", b1_grid),
    ("B3", b3_bo),
    ("B4", b4_cma),
    ("B5", b5_labels),
    ("M1", m1_sim_filter),
)
# B5 and M1 get identical user slices and delayed-V observations. Their only
# difference is select-best-measured versus fit-and-extrapolate.
METHOD_SEED = {"B0": 0, "B1": 1, "B3": 3, "B4": 4, "B5": 5, "M1": 5}


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
            "n_labels": len(harness.labeled()),
            "proxy": scored.proxy,
            "value": scored.value,
            "unsafe": scored.unsafe,
            "bait_share": scored.bait_share,
            "regret": normalised_regret(scored.value, v_star, v0, scored.unsafe),
            "weights": commit.weights.tolist(),
            "chosen_by": commit.chosen_by,
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
        "methods": rows,
        "exploratory": True,
    }


def main() -> int:
    spec = PilotSpec()
    params = WorldParams()
    seeds = [spec.seed0 + i for i in range(spec.n_seeds)]
    runs = []
    for s in seeds:
        print(f"  seed {s} …", flush=True)
        runs.append(run_seed(s, spec, params))
        v = runs[-1]["methods"]
        print(
            f"    B0={v['B0']['value']:.3f}  B1={v['B1']['value']:.3f}  "
            f"B3={v['B3']['value']:.3f}  B4={v['B4']['value']:.3f}  "
            f"B5={v['B5']['value']:.3f}  M1={v['M1']['value']:.3f}  ({v['M1']['chosen_by']})",
            flush=True,
        )
    regrets = {name: [run["methods"][name]["regret"] for run in runs] for name, _ in METHODS}
    mean_value = {
        name: float(np.mean([run["methods"][name]["value"] for run in runs])) for name, _ in METHODS
    }
    walkbacks = {
        name: int(sum(run["methods"][name]["walkback"] for run in runs)) for name, _ in METHODS
    }
    paired_m1_b5 = [
        run["methods"]["M1"]["value"] - run["methods"]["B5"]["value"] for run in runs
    ]
    summary = {
        "phase": "2-exploratory",
        "note": (
            "Not confirmatory. Same 5 pilot seeds as phase 1. "
            "M1 vs B5 is the load-bearing comparison: identical budget, identical "
            "access to delayed V, surrogate vs no surrogate."
        ),
        "spec": {
            "budget_user_days": spec.budget_user_days,
            "m1_label_users": spec.m1_label_users,
            "m1_confirm_users": spec.m1_confirm_users,
        },
        "runs": runs,
        "mean_regret": {k: float(np.mean(v)) for k, v in regrets.items()},
        "mean_value": mean_value,
        "walkbacks": walkbacks,
        "m1_vs_b0_value": mean_value["M1"] - mean_value["B0"],
        "m1_vs_b3_value": mean_value["M1"] - mean_value["B3"],
        "m1_vs_b4_value": mean_value["M1"] - mean_value["B4"],
        "m1_vs_b5_value": mean_value["M1"] - mean_value["B5"],
        "m1_vs_b5_paired": paired_m1_b5,
        "m1_beats_b5_on_seeds": int(sum(d > 0 for d in paired_m1_b5)),
    }
    out = Path(__file__).resolve().parents[2] / "analysis" / "phase2_m1_exploratory.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        plot_committed_v(
            summary,
            out.parent / "figures" / "p2_m1_committed_v.png",
            names=["B0", "B1", "B3", "B4", "B5", "M1"],
            title="Exploratory · M1 vs baselines · committed V (not confirmatory)",
        )
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    print("phase 2 exploratory  (not confirmatory)")
    for name, _ in METHODS:
        rs = regrets[name]
        print(
            f"  {name}  mean V={mean_value[name]:.3f}  mean R={np.mean(rs):.3f}  "
            f"walkbacks={walkbacks[name]}/5"
        )
    print(f"  M1 − B0  ΔV={summary['m1_vs_b0_value']:+.3f}")
    print(f"  M1 − B3  ΔV={summary['m1_vs_b3_value']:+.3f}")
    print(f"  M1 − B4  ΔV={summary['m1_vs_b4_value']:+.3f}")
    print(
        f"  M1 − B5  ΔV={summary['m1_vs_b5_value']:+.3f}   "
        f"(M1 wins {summary['m1_beats_b5_on_seeds']}/5 seeds)   <- the real test"
    )
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
