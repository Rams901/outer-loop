"""P2 fidelity sweep: M1 against a dialled simulator vs B5.

Not confirmatory H3. B5 does not use the sim, so it is a horizontal bar.
M1 searches SimWorld(φ) for free and confirms on TrueWorld. φ=1 is true V;
φ=0 is bait share.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_opt.config import WorldParams
from rl_opt.fidelity import PHI_GRID, FidelitySim, m1_against_sim
from rl_opt.harness import Budget, Harness, Oracle
from rl_opt.search import b0_default, b5_labels
from rl_opt.space import PilotSpec
from rl_opt.world import TrueWorld


def _crossover(phis: list[float], m1: list[float], b5: float) -> float | None:
    """Smallest φ where M1's mean V meets or exceeds B5, linearly interpolated."""
    diffs = [v - b5 for v in m1]
    if diffs[0] >= 0:
        return float(phis[0])
    for i in range(len(diffs) - 1):
        if diffs[i] < 0 <= diffs[i + 1]:
            span = diffs[i + 1] - diffs[i]
            if abs(span) < 1e-12:
                return float(phis[i + 1])
            t = -diffs[i] / span
            return float(phis[i] + t * (phis[i + 1] - phis[i]))
    return None


def run_seed(world_seed: int, spec: PilotSpec, params: WorldParams, phis: tuple[float, ...]) -> dict:
    world = TrueWorld.generate(params, seed=world_seed)
    oracle = Oracle(world, rollout_seed=spec.eval_rollout_seed)
    b0 = b0_default(
        Harness(world, Budget(spec.budget_user_days), np.random.default_rng(0)),
        spec,
    )
    b5_harness = Harness(
        world,
        Budget(spec.budget_user_days),
        np.random.default_rng(world_seed * 1009 + 5),
    )
    b5 = b5_labels(b5_harness, spec)
    b0_v = oracle.value(b0.weights).value
    b5_scored = oracle.value(b5.weights)
    m1_rows = {}
    for phi in phis:
        print(f"    φ={phi:.2f} …", flush=True)
        sim = FidelitySim(world, phi=phi, n_users=spec.fidelity_sim_users)
        harness = Harness(
            world,
            Budget(spec.budget_user_days),
            np.random.default_rng(world_seed * 1009 + 5),
        )
        commit = m1_against_sim(harness, spec, sim)
        scored = oracle.value(commit.weights)
        m1_rows[str(phi)] = {
            "phi": phi,
            "value": scored.value,
            "proxy": scored.proxy,
            "bait_share": scored.bait_share,
            "unsafe": scored.unsafe,
            "spent": commit.spent,
            "n_labels": len(harness.labeled()),
            "chosen_by": commit.chosen_by,
            "weights": commit.weights.tolist(),
            "beats_b5": bool(scored.value > b5_scored.value),
        }
    return {
        "world_seed": world_seed,
        "b0_value": b0_v,
        "b5_value": b5_scored.value,
        "b5_bait_share": b5_scored.bait_share,
        "m1": m1_rows,
    }


def main() -> int:
    spec = PilotSpec(m1_sim_draws=128)
    params = WorldParams()
    seeds = [spec.seed0 + i for i in range(spec.n_seeds)]
    runs = []
    for seed in seeds:
        print(f"  seed {seed} …", flush=True)
        runs.append(run_seed(seed, spec, params, PHI_GRID))
        m1 = runs[-1]["m1"]
        bits = "  ".join(f"φ{phi:g}={m1[str(phi)]['value']:.3f}" for phi in PHI_GRID)
        print(f"    B0={runs[-1]['b0_value']:.3f}  B5={runs[-1]['b5_value']:.3f}  {bits}", flush=True)

    mean_b0 = float(np.mean([run["b0_value"] for run in runs]))
    mean_b5 = float(np.mean([run["b5_value"] for run in runs]))
    mean_m1 = [float(np.mean([run["m1"][str(phi)]["value"] for run in runs])) for phi in PHI_GRID]
    wins = {
        str(phi): int(sum(run["m1"][str(phi)]["beats_b5"] for run in runs)) for phi in PHI_GRID
    }
    cross = _crossover(list(PHI_GRID), mean_m1, mean_b5)
    summary = {
        "phase": "2-fidelity-exploratory",
        "note": (
            "Not confirmatory H3. Scalar dial mixes true V with bait share. "
            "M1 searches the sim free and confirms on TrueWorld. B5 is unchanged."
        ),
        "phi_grid": list(PHI_GRID),
        "spec": {
            "budget_user_days": spec.budget_user_days,
            "m1_label_users": spec.m1_label_users,
            "m1_sim_draws": spec.m1_sim_draws,
            "fidelity_sim_users": spec.fidelity_sim_users,
        },
        "runs": runs,
        "mean_b0": mean_b0,
        "mean_b5": mean_b5,
        "mean_m1": {str(phi): v for phi, v in zip(PHI_GRID, mean_m1)},
        "m1_beats_b5_on_seeds": wins,
        "crossover_phi_vs_b5": cross,
    }
    out = Path(__file__).resolve().parents[2] / "analysis" / "phase2_fidelity.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        from rl_opt.figures import plot_fidelity_curve

        plot_fidelity_curve(summary, out.parent / "figures" / "p2_fidelity.png")
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    print("phase 2 fidelity  (exploratory, not H3)")
    print(f"  B0 mean V={mean_b0:.3f}   B5 mean V={mean_b5:.3f}")
    for phi, value in zip(PHI_GRID, mean_m1):
        print(f"  M1 φ={phi:.2f}  mean V={value:.3f}  beats B5 on {wins[str(phi)]}/5")
    if cross is None:
        print("  no crossover: M1 never reaches B5 on this dial")
    else:
        print(f"  crossover vs B5 at φ≈{cross:.2f}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
