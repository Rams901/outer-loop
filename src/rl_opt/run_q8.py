"""Q8 taxonomy sweep (exploratory). Not confirmatory H3.

Four wrongness axes, same M1-search-then-confirm vs B5 protocol as docs/12.
φ grid is coarse (0, 0.5, 1) so the extra axes fit on a laptop next to the
already-run bait-blindness sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_opt.config import WorldParams
from rl_opt.fidelity import Q8_AXES, Q8_PHI_GRID, FidelitySim, m1_against_sim
from rl_opt.harness import Budget, Harness, Oracle
from rl_opt.run_fidelity import _crossover
from rl_opt.search import b0_default, b5_labels
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld


def run_seed(
    world_seed: int,
    spec: PilotSpec,
    params: WorldParams,
    axes: tuple[str, ...],
    phis: tuple[float, ...],
) -> dict:
    world = TrueWorld.generate(params, seed=world_seed)
    oracle = Oracle(world, rollout_seed=spec.eval_rollout_seed)
    b0 = oracle.value(
        b0_default(Harness(world, Budget(spec.budget_user_days), np.random.default_rng(0)), spec).weights
    )
    b5_harness = Harness(
        world, Budget(spec.budget_user_days), np.random.default_rng(world_seed * 1009 + 5)
    )
    b5 = oracle.value(b5_labels(b5_harness, spec).weights)
    axes_out = {}
    for axis in axes:
        rows = {}
        for phi in phis:
            print(f"    {axis} φ={phi:.2f} …", flush=True)
            sim = FidelitySim(world, phi=phi, n_users=spec.fidelity_sim_users, axis=axis)
            harness = Harness(
                world, Budget(spec.budget_user_days), np.random.default_rng(world_seed * 1009 + 5)
            )
            commit = m1_against_sim(harness, spec, sim)
            scored = oracle.value(commit.weights)
            rows[str(phi)] = {
                "phi": phi,
                "value": scored.value,
                "bait_share": scored.bait_share,
                "proxy": scored.proxy,
                "unsafe": scored.unsafe,
                "chosen_by": commit.chosen_by,
                "beats_b5": bool(scored.value > b5.value),
            }
        axes_out[axis] = rows
    return {
        "world_seed": world_seed,
        "b0_value": b0.value,
        "b5_value": b5.value,
        "axes": axes_out,
    }


def main() -> int:
    spec = PilotSpec(m1_sim_draws=64)
    params = WorldParams()
    seeds = [spec.seed0 + i for i in range(spec.n_seeds)]
    runs = []
    for seed in seeds:
        print(f"  seed {seed} …", flush=True)
        runs.append(run_seed(seed, spec, params, Q8_AXES, Q8_PHI_GRID))

    mean_b0 = float(np.mean([run["b0_value"] for run in runs]))
    mean_b5 = float(np.mean([run["b5_value"] for run in runs]))
    by_axis = {}
    for axis in Q8_AXES:
        mean_m1 = [
            float(np.mean([run["axes"][axis][str(phi)]["value"] for run in runs]))
            for phi in Q8_PHI_GRID
        ]
        wins = {
            str(phi): int(sum(run["axes"][axis][str(phi)]["beats_b5"] for run in runs))
            for phi in Q8_PHI_GRID
        }
        by_axis[axis] = {
            "mean_m1": {str(phi): v for phi, v in zip(Q8_PHI_GRID, mean_m1)},
            "m1_beats_b5_on_seeds": wins,
            "crossover_phi_vs_b5": _crossover(list(Q8_PHI_GRID), mean_m1, mean_b5),
        }
    summary = {
        "phase": "2-q8-exploratory",
        "note": (
            "Not confirmatory H3. Four Q8 axes, coarse φ grid, same 5 exploratory "
            "seeds as docs/12. Supply-feedback axis omitted (off in this world)."
        ),
        "phi_grid": list(Q8_PHI_GRID),
        "axes": list(Q8_AXES),
        "mean_b0": mean_b0,
        "mean_b5": mean_b5,
        "by_axis": by_axis,
        "runs": runs,
    }
    out = Path(__file__).resolve().parents[2] / "analysis" / "phase2_q8.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        from rl_opt.figures import plot_q8_axes

        plot_q8_axes(summary, out.parent / "figures" / "p2_q8.png")
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    print("Q8 taxonomy  (exploratory, not H3)")
    print(f"  B0={mean_b0:.3f}  B5={mean_b5:.3f}")
    for axis, row in by_axis.items():
        cross = row["crossover_phi_vs_b5"]
        bits = "  ".join(
            f"φ{phi:g}={row['mean_m1'][str(phi)]:.3f}({row['m1_beats_b5_on_seeds'][str(phi)]}/5)"
            for phi in Q8_PHI_GRID
        )
        print(f"  {axis:8s} {bits}  cross={cross if cross is None else f'{cross:.2f}'}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
