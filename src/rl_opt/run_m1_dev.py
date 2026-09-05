"""Choose the single M1 retune on development seeds, never evaluation seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_opt.config import WorldParams
from rl_opt.harness import Budget, Harness, Oracle
from rl_opt.m1 import (
    m1_gp_lcb_half,
    m1_gp_lcb_one,
    m1_gp_mean,
    m1_ridge_no_confirm,
)
from rl_opt.search import b5_labels
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld

DEV_SEEDS = (20, 21, 22, 23, 24)
METHODS = (
    ("B5", b5_labels),
    ("ridge", m1_ridge_no_confirm),
    ("gp_mean", m1_gp_mean),
    ("gp_lcb_0.5", m1_gp_lcb_half),
    ("gp_lcb_1.0", m1_gp_lcb_one),
)


def main() -> int:
    spec = PilotSpec()
    rows = []
    for world_seed in DEV_SEEDS:
        print(f"  dev seed {world_seed} …", flush=True)
        world = TrueWorld.generate(WorldParams(), seed=world_seed)
        oracle = Oracle(world, rollout_seed=spec.eval_rollout_seed)
        methods = {}
        for name, method in METHODS:
            # Identical RNG means identical user slices and observations.
            harness = Harness(
                world,
                Budget(spec.budget_user_days),
                np.random.default_rng(world_seed * 1009 + 5),
            )
            commit = method(harness, spec)
            result = oracle.value(commit.weights)
            methods[name] = {
                "value": result.value,
                "bait_share": result.bait_share,
                "unsafe": result.unsafe,
                "spent": commit.spent,
                "n_labels": len(harness.labeled()),
                "chosen_by": commit.chosen_by,
                "weights": commit.weights.tolist(),
            }
        rows.append({"world_seed": world_seed, "methods": methods})
        print(
            "    "
            + "  ".join(f"{name}={methods[name]['value']:.3f}" for name, _ in METHODS),
            flush=True,
        )

    means = {
        name: float(np.mean([row["methods"][name]["value"] for row in rows]))
        for name, _ in METHODS
    }
    candidates = {name: value for name, value in means.items() if name != "B5"}
    selected = max(candidates, key=candidates.get)
    summary = {
        "phase": "m1-retune-development-only",
        "development_seeds": list(DEV_SEEDS),
        "evaluation_seeds_not_used": [10, 11, 12, 13, 14],
        "selection_rule": "highest mean committed V; freeze before evaluation",
        "rows": rows,
        "mean_value": means,
        "selected": selected,
        "selected_vs_b5": means[selected] - means["B5"],
    }
    out = Path(__file__).resolve().parents[2] / "analysis" / "m1_retune_dev.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  selected {selected}: mean V={means[selected]:.3f}")
    print(f"  B5 mean V={means['B5']:.3f}; delta={summary['selected_vs_b5']:+.3f}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
