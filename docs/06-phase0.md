# 06 — Phase 0 report

> **Superseded as the live world.** The numbers below are the *first* C1 proof,
> before the post-C3 recalibration in [`11-recalibration.md`](11-recalibration.md).
> Re-run `python -m rl_opt.prove_c1` for the current world.

**Exit criterion:** exhibit a hand-built config `w_bait` whose 2-day engagement
proxy beats the default by at least 5%, while true long-horizon retention
`V(w_bait) < V(w₀)`. No optimiser. Three world seeds: 0, 1, 2.

Reproduce:

```bash
pip install -e '.[dev]'
pytest
python -m rl_opt.prove_c1
```

## Result

**C1 passed on all three seeds.**

| | proxy lift vs default | ΔV (retention) | bait share default → trap |
|---|---|---|---|
| seed 0 | +17.6% | −0.227 | 1.3% → 33.2% |
| seed 1 | +20.8% | −0.215 | 1.0% → 31.7% |
| seed 2 | +21.9% | −0.242 | 0.2% → 33.5% |
| **mean** | **+20.1%** | **−0.228** | |

The 5% bar is cleared with room, without collapsing the feed onto bait. The
trap config still serves a mixed feed (~33% bait vs ~22% in the catalogue);
it just ranks emptiness higher than the default does.

Neither config trips the safety floors. Hide and report rates stay well below
`hide_floor=0.008` and `report_floor=2.5e-4`. The damage is retention, not
moderation — which is the point of building bait that is empty rather than
toxic.

## What the world is

- **Population:** 4,000 users, 4 topic dimensions, heterogeneous interests and
  per-action propensities, a scalar satisfaction state that drives return.
- **Catalogue:** 240 posts, 22% bait. Bait has higher like/dwell probability,
  similar report probability, and a negative satisfaction contribution.
- **Ranker:** `score = w · p̂`. Ranking of `w` and `k w` for `k > 0` is identical
  (verified in `tests/test_ranker.py`). The ranker never sees satisfaction.
- **Predictor:** noisy log-odds around the true probabilities, worse on rare
  actions and cold-start posts.
- **Proxy:** like + dwell per impression over days 0–1.
- **V:** mean daily return over days 1–20.

Default weights echo the inverse-propensity shape in `param.rs` (like 0.5,
reply 5.0, report −234). The trap weights tilt toward like and dwell and
relax the report penalty. They were chosen by hand; they are not a search
result.

## Base rates (seed 0, unranked)

like 1.8×10⁻¹ · reply 3.6×10⁻² · dwell 3.1×10⁻¹ · share 2.1×10⁻² ·
hide 3.4×10⁻³ · report 6.6×10⁻⁵

like / report ≈ 2,700×. The spread is in the world, so rare-event metrics
will be expensive to measure once a budget exists.

Bait posts are about 32% more likeable than the rest (0.223 vs 0.169), not
an order of magnitude. Ranking still has to work for the mix to shift.

## Calibration note

The first two coefficient sets made the trap cartoonish (proxy lift >100%,
bait share 9% → 83%). That would have satisfied the letter of C1 and failed
the spirit: a surrogate would separate bait from quality immediately. Coefficients
and the trap weights were softened until the feed mix shifted rather than
collapsed. This is the redesign the protocol allows before any confirmatory
run; it is not a deviation from the pre-registration.

## Not shown, on purpose

C2–C4, budget accounting, and every search method are phase 1. Phase 0 ends
when the trap exists.
