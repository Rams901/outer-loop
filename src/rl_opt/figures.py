"""Figures from traces and daily series. Optional matplotlib extra."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rl_opt.config import ACTIONS
from rl_opt.metrics import RolloutResult

VIOLET = "#7C5CFF"
ROSE = "#FF5470"
AMBER = "#FFC24B"
BLUE = "#4DA3FF"
TEXT = "#EEF1F8"
DIM = "#8494B4"
BG = "#070A14"
SURFACE = "#12182A"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=DIM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2A3350")
    ax.spines["left"].set_color("#2A3350")
    ax.yaxis.label.set_color(DIM)
    ax.xaxis.label.set_color(DIM)
    ax.title.set_color(TEXT)


def disputed_candidates(trace: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Items shown in either feed, ordered by rank disagreement."""
    shown = [c for c in trace["candidates"] if c["shown"]["default"] or c["shown"]["engagement"]]
    shown.sort(key=lambda c: abs(c["rank"]["default"] - c["rank"]["engagement"]), reverse=True)
    return shown[:limit]


def plot_score_terms(trace: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = disputed_candidates(trace)
    labels = [f"p{c['post_id']}\n{'bait' if c['bait'] > 0.5 else 'ok'}" for c in rows]
    x = np.arange(len(rows))
    width = 0.38
    def_totals = [c["score_terms"]["default"]["total"] for c in rows]
    eng_totals = [c["score_terms"]["engagement"]["total"] for c in rows]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    _style(ax)
    ax.bar(x - width / 2, def_totals, width, color=VIOLET, label="default w")
    ax.bar(x + width / 2, eng_totals, width, color=ROSE, label="engagement w")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("score  =  w · p̂")
    ax.set_title("Same candidates, two weight vectors")
    ax.legend(facecolor=SURFACE, edgecolor="#2A3350", labelcolor=TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_horizon(default: RolloutResult, engagement: RolloutResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    days = np.arange(len(default.daily_return))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    series = [
        (axes[0], "daily return (V's ingredients)", default.daily_return, engagement.daily_return),
        (axes[1], "bait share of impressions", default.daily_bait_share, engagement.daily_bait_share),
        (axes[2], "like+dwell rate that day", default.daily_engagement, engagement.daily_engagement),
    ]
    for ax, title, a, b in series:
        _style(ax)
        ax.plot(days, a, color=VIOLET, lw=2, label="default")
        ax.plot(days, b, color=ROSE, lw=2, label="engagement")
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(facecolor=SURFACE, edgecolor="#2A3350", labelcolor=TEXT, fontsize=8)
    axes[2].set_xlabel("day")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_committed_v(
    summary: dict,
    path: Path,
    *,
    names: list[str] | None = None,
    title: str = "Committed config, true objective",
) -> None:
    import matplotlib.pyplot as plt

    names = names or list(summary["mean_value"].keys())
    v = [summary["mean_value"][n] for n in names]
    palette = {
        "B0": VIOLET,
        "B1": ROSE,
        "B2": "#5B6B8C",
        "B3": ROSE,
        "B4": AMBER,
        "B5": "#3FBF8F",
        "M1": BLUE,
        "U1": TEXT,
    }
    colors = [palette.get(n, DIM) for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(ax)
    ax.bar(names, v, color=colors)
    ax.set_ylabel("mean V (retention)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fidelity_curve(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    phis = [float(p) for p in summary["phi_grid"]]
    m1 = [summary["mean_m1"][str(p)] for p in summary["phi_grid"]]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _style(ax)
    ax.plot(phis, m1, color=BLUE, lw=2.2, marker="o", label="M1 (sim search)")
    ax.axhline(summary["mean_b5"], color="#3FBF8F", lw=1.8, ls="--", label="B5 (11 labels)")
    ax.axhline(summary["mean_b0"], color=VIOLET, lw=1.8, ls=":", label="B0 (default)")
    cross = summary.get("crossover_phi_vs_b5")
    if cross is not None:
        ax.axvline(cross, color=AMBER, lw=1.2, ls="-.", label=f"crossover φ≈{cross:.2f}")
    ax.set_xlabel("simulator fidelity φ  (0 = bait, 1 = true V)")
    ax.set_ylabel("mean V (retention)")
    ax.set_title("Exploratory P2 · fidelity sweep  (not confirmatory H3)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(facecolor=SURFACE, edgecolor="#2A3350", labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_q8_axes(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    phis = [float(p) for p in summary["phi_grid"]]
    palette = {"bait": BLUE, "proxy": ROSE, "stale": AMBER, "rare": VIOLET}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _style(ax)
    for axis, row in summary["by_axis"].items():
        ys = [row["mean_m1"][str(p)] for p in summary["phi_grid"]]
        ax.plot(phis, ys, color=palette.get(axis, DIM), lw=2, marker="o", label=axis)
    ax.axhline(summary["mean_b5"], color="#3FBF8F", lw=1.6, ls="--", label="B5")
    ax.axhline(summary["mean_b0"], color=TEXT, lw=1.2, ls=":", label="B0")
    ax.set_xlabel("simulator fidelity φ")
    ax.set_ylabel("mean V (retention)")
    ax.set_title("Exploratory Q8 · four wrongness axes  (not H3)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(facecolor=SURFACE, edgecolor="#2A3350", labelcolor=TEXT, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pilot_bars(summary: dict, path: Path) -> None:
    plot_committed_v(
        summary,
        path,
        names=["B0", "B1", "B2", "B3", "B4"],
        title="Phase 1 pilot · committed config, true objective",
    )


def stacked_terms_table(trace: dict[str, Any]) -> list[list[str]]:
    """Plain table for docs / HTML; not a figure."""
    header = ["post", "kind", "rank_def", "rank_eng", *ACTIONS, "total_def", "total_eng"]
    rows = [header]
    for c in disputed_candidates(trace):
        kind = "bait" if c["bait"] > 0.5 else "ok"
        d = c["score_terms"]["default"]
        e = c["score_terms"]["engagement"]
        rows.append(
            [
                str(c["post_id"]),
                kind,
                str(c["rank"]["default"]),
                str(c["rank"]["engagement"]),
                *[f"{d[a]:.3f}" for a in ACTIONS],
                f"{d['total']:.3f}",
                f"{e['total']:.3f}",
            ]
        )
    return rows
