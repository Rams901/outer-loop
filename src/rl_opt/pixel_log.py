"""Build a 21-day pixel log from a real TrueWorld rollout.

The pygame viewer reads this. It does not re-rank. Layout by topic is a legend,
not a city the ranker walks through — there is no follow-graph in this world;
in-interest vs OON is relevance vs the catalogue median.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.world import TrueWorld

ROOT = Path(__file__).resolve().parents[2]


def pick_watch_ids(world: TrueWorld, n_watch: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(int(n_watch), world.params.n_users)
    topics = world.primary_topic()
    # Stratify by topic so the grid is a readable legend, not a random soup.
    chosen: list[int] = []
    per = max(1, n // world.params.n_topics)
    for t in range(world.params.n_topics):
        pool = np.flatnonzero(topics == t)
        if pool.size == 0:
            continue
        take = min(per, int(pool.size))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    leftover = [i for i in range(world.params.n_users) if i not in set(chosen)]
    rng.shuffle(leftover)
    while len(chosen) < n and leftover:
        chosen.append(leftover.pop())
    return np.asarray(chosen[:n], dtype=np.int64)


def record_pixel_logs(
    world_seed: int = 0,
    n_watch: int = 96,
    rollout_seed: int = 7,
    params: WorldParams | None = None,
) -> dict:
    params = params or WorldParams()
    world = TrueWorld.generate(params, seed=world_seed)
    watch = pick_watch_ids(world, n_watch)
    logs = {}
    for name, weights in (("default", DEFAULT_WEIGHTS), ("engagement", ENGAGEMENT_WEIGHTS)):
        world.rollout(weights, seed=rollout_seed, watch_ids=watch)
        assert world.last_pixel_log is not None
        logs[name] = world.last_pixel_log
        logs[name]["config"] = name
    payload = {
        "schema": "rl_opt.pixel.v1",
        "world_seed": world_seed,
        "n_watch": int(watch.size),
        "configs": logs,
    }
    out = ROOT / "analysis" / "traces" / "pixel_cohort.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n")
    return payload


def main() -> int:
    payload = record_pixel_logs()
    print(f"  watch={payload['n_watch']}  wrote analysis/traces/pixel_cohort.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
