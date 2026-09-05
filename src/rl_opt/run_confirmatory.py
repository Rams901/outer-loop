"""Confirmatory H1: frozen M1 vs B5 on stage-2 seeds.

No interim look at the primary comparison in this file's control flow: the full
N is scheduled up front, results are checkpointed per seed so a crash can resume
the same seeds, and nothing is dropped.
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
from rl_opt.search import b0_default, b5_labels
from rl_opt.stage2 import Stage2Lock, build_lock, write_lock
from rl_opt.stats import bootstrap_median_ci, wilcoxon_signed_rank
from rl_opt.world import TrueWorld

METHODS = (
    ("B0", b0_default),
    ("B5", b5_labels),
    ("M1", m1_sim_filter),
)
METHOD_SEED = {"B0": 0, "B5": 5, "M1": 5}

OUT = Path(__file__).resolve().parents[2] / "analysis" / "confirmatory_h1.json"


def run_seed(world_seed: int, lock: Stage2Lock, params: WorldParams) -> dict:
    spec = lock.spec()
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
        "w_star": w_star.tolist(),
        "methods": rows,
        "confirmatory": True,
    }


def _summarise(runs: list[dict], lock: Stage2Lock) -> dict:
    names = [name for name, _ in METHODS]
    regrets = {name: [run["methods"][name]["regret"] for run in runs] for name in names}
    mean_value = {
        name: float(np.mean([run["methods"][name]["value"] for run in runs])) for name in names
    }
    paired_r = [
        run["methods"]["M1"]["regret"] - run["methods"]["B5"]["regret"] for run in runs
    ]
    paired_v = [
        run["methods"]["M1"]["value"] - run["methods"]["B5"]["value"] for run in runs
    ]
    primary = wilcoxon_signed_rank(paired_r)
    ci = bootstrap_median_ci(paired_r, seed=1)
    return {
        "phase": "confirmatory-h1",
        "lock": {
            "n": lock.n,
            "seeds": list(lock.confirmatory_seeds),
            "frozen_m1": lock.frozen_m1,
            "mde": lock.mde,
        },
        "n_completed": len(runs),
        "runs": runs,
        "mean_regret": {k: float(np.mean(v)) for k, v in regrets.items()},
        "mean_value": mean_value,
        "walkbacks": {
            name: int(sum(run["methods"][name]["walkback"] for run in runs)) for name in names
        },
        "primary": {
            "comparison": "M1 − B5 in R",
            "paired": paired_r,
            "paired_value": paired_v,
            "wilcoxon": primary,
            "bootstrap_median_ci": ci,
            "h1_reject_null": bool(primary["p_value"] < 0.05),
            "h1_m1_better": bool(primary["median"] < 0 and primary["p_value"] < 0.05),
        },
        "secondary": {
            "note": "H1-original / H2 / H4 require B1 and B3; this run is primary H1 vs B5 only."
        },
    }


def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("runs") or [])


def main() -> int:
    lock_path = OUT.parent / "stage2_lock.json"
    lock = write_lock() if not lock_path.exists() else build_lock()
    if not lock_path.exists():
        write_lock()
    params = WorldParams()
    done = {run["world_seed"]: run for run in _load_checkpoint(OUT)}
    for seed in lock.confirmatory_seeds:
        if seed in done:
            print(f"  seed {seed} (checkpoint)", flush=True)
            continue
        print(f"  seed {seed} …", flush=True)
        row = run_seed(seed, lock, params)
        done[seed] = row
        ordered = [done[s] for s in lock.confirmatory_seeds if s in done]
        summary = _summarise(ordered, lock)
        summary["status"] = "running" if len(ordered) < lock.n else "complete"
        OUT.write_text(json.dumps(summary, indent=2) + "\n")
        v = row["methods"]
        print(
            f"    B0={v['B0']['value']:.3f}  B5={v['B5']['value']:.3f}  "
            f"M1={v['M1']['value']:.3f}",
            flush=True,
        )
    missing = [s for s in lock.confirmatory_seeds if s not in done]
    if missing:
        print(f"  incomplete, missing seeds {missing}")
        return 1
    summary = _summarise([done[s] for s in lock.confirmatory_seeds], lock)
    summary["status"] = "complete"
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        plot_committed_v(
            summary,
            OUT.parent / "figures" / "h1_confirmatory.png",
            names=["B0", "B5", "M1"],
            title="Confirmatory H1 · committed V · M1 vs B5",
        )
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    primary = summary["primary"]
    print("confirmatory H1  (M1 vs B5 in R)")
    print(f"  N={lock.n}  status={summary['status']}")
    print(
        f"  mean V  B5={summary['mean_value']['B5']:.3f}  "
        f"M1={summary['mean_value']['M1']:.3f}"
    )
    print(
        f"  Wilcoxon median ΔR={primary['wilcoxon']['median']:+.3f}  "
        f"p={primary['wilcoxon']['p_value']:.4f}  "
        f"CI [{primary['bootstrap_median_ci']['ci_low']:+.3f}, "
        f"{primary['bootstrap_median_ci']['ci_high']:+.3f}]"
    )
    print(f"  H1 M1 better (median<0 and p<0.05): {primary['h1_m1_better']}")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
