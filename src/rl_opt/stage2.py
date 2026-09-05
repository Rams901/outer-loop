"""Stage-2 lock. Executing the pre-registered sample-size procedure.

Pilot seeds 10–14 and development seeds 20–24 are never reused. Confirmatory
H1 is frozen M1 (GP LCB β=1) vs B5 at the delayed-label design that the one
allowed retune selected — not a second retune, and not logs-under-w₀ (that
deviation is in the pre-reg log).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.space import PilotSpec
from rl_opt.stats import MDE, sample_size_from_pilot_sd
from rl_opt.world import (
    RETURN_K,
    SAT_BAIT,
    SAT_INIT,
    SAT_QUALITY,
    SAT_STEP,
)

ROOT = Path(__file__).resolve().parents[2]
PILOT_JSON = ROOT / "analysis" / "phase2_m1_exploratory.json"

PILOT_SEEDS = (10, 11, 12, 13, 14)
DEV_SEEDS = (20, 21, 22, 23, 24)
CONFIRMATORY_SEED0 = 30
BUDGET_USER_DAYS = 24_000
M1_LABEL_USERS = 100
HORIZON_DAYS = 21
# 11 full-horizon labels × 100 users. Shared by B5 and fitted M1.
L_LABELS = BUDGET_USER_DAYS // (M1_LABEL_USERS * HORIZON_DAYS)
TUNING_BUDGET = 4  # configs evaluated on DEV_SEEDS; already spent; frozen.


@dataclass(frozen=True)
class Stage2Lock:
    n: int
    confirmatory_seeds: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    budget_user_days: int
    m1_label_users: int
    l_labels: int
    l_user_days: int
    horizon_days: int
    mde: float
    frozen_m1: str
    power: dict
    world: dict
    note: str

    def spec(self) -> PilotSpec:
        return PilotSpec(
            budget_user_days=self.budget_user_days,
            n_seeds=self.n,
            seed0=self.confirmatory_seeds[0],
            m1_label_users=self.m1_label_users,
        )


def _paired_regret(path: Path = PILOT_JSON) -> list[float]:
    payload = json.loads(path.read_text())
    diffs = []
    for run in payload["runs"]:
        methods = run["methods"]
        diffs.append(float(methods["M1"]["regret"]) - float(methods["B5"]["regret"]))
    return diffs


def _world_freeze() -> dict:
    params = asdict(WorldParams())
    params.update(
        {
            "SAT_STEP": SAT_STEP,
            "SAT_INIT": SAT_INIT,
            "SAT_QUALITY": SAT_QUALITY,
            "SAT_BAIT": SAT_BAIT,
            "RETURN_K": RETURN_K,
            "DEFAULT_WEIGHTS": DEFAULT_WEIGHTS.tolist(),
            "ENGAGEMENT_WEIGHTS": ENGAGEMENT_WEIGHTS.tolist(),
        }
    )
    return params


def build_lock(path: Path = PILOT_JSON) -> Stage2Lock:
    diffs = _paired_regret(path)
    sd = float(np.std(diffs, ddof=1))
    power = sample_size_from_pilot_sd(sd, mde=MDE)
    n = int(power["n"])
    seeds = tuple(CONFIRMATORY_SEED0 + i for i in range(n))
    return Stage2Lock(
        n=n,
        confirmatory_seeds=seeds,
        pilot_seeds=PILOT_SEEDS,
        development_seeds=DEV_SEEDS,
        budget_user_days=BUDGET_USER_DAYS,
        m1_label_users=M1_LABEL_USERS,
        l_labels=L_LABELS,
        l_user_days=L_LABELS * M1_LABEL_USERS * HORIZON_DAYS,
        horizon_days=HORIZON_DAYS,
        mde=MDE,
        frozen_m1="m1_gp_lcb_one",
        power=power,
        world=_world_freeze(),
        note=(
            "N from paired (M1−B5) regret sd on exploratory seeds 10–14. "
            "Those seeds are not confirmatory. Realistic fidelity for this lock "
            "is the shared 11-label delayed-V design, not logs under w0 only."
        ),
    )


def lock_dict(lock: Stage2Lock) -> dict:
    return {
        "stage": 2,
        "locked": True,
        "n": lock.n,
        "confirmatory_seeds": list(lock.confirmatory_seeds),
        "pilot_seeds": list(lock.pilot_seeds),
        "development_seeds": list(lock.development_seeds),
        "budget_user_days": lock.budget_user_days,
        "m1_label_users": lock.m1_label_users,
        "l_labels": lock.l_labels,
        "l_user_days": lock.l_user_days,
        "horizon_days": lock.horizon_days,
        "mde": lock.mde,
        "frozen_m1": lock.frozen_m1,
        "power": lock.power,
        "world": lock.world,
        "note": lock.note,
    }


def write_lock(out: Path | None = None) -> Stage2Lock:
    lock = build_lock()
    target = out or (ROOT / "analysis" / "stage2_lock.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(lock_dict(lock), indent=2) + "\n")
    return lock


def main() -> int:
    lock = write_lock()
    print("stage 2 lock")
    print(f"  pilot (M1−B5) R sd={lock.power['pilot_sd']:.3f}")
    print(f"  paired-t n={lock.power['n_paired_t']}  Wilcoxon n={lock.power['n_wilcoxon']}")
    print(f"  N={lock.n}  seeds {lock.confirmatory_seeds[0]}–{lock.confirmatory_seeds[-1]}")
    print(f"  L={lock.l_labels} delayed-V labels ({lock.l_user_days} user-days)")
    print(f"  frozen M1={lock.frozen_m1}")
    print(f"  wrote {ROOT / 'analysis' / 'stage2_lock.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
