# 10 — Strengthened oracle (step 0 diagnostic)

**Question:** is `V* − V₀ ≈ 0.0075` a fact about the world, or an artifact of
U1 making 30 oracle calls at a single noisy rollout seed, one of which was a
config we wrote by hand?

**Protocol.** 145 candidates per seed (hand pool + 32 random + 8×12 ES on
log-weights). Each search evaluation averages **2** rollout seeds. The top 12
and the default are then confirmed on **8 held-out** rollout seeds. Search and
confirm seeds are disjoint, so reported `V*` is not the maximum of the search
noise. 394 oracle calls per world seed.

```bash
python -m rl_opt.run_u1_strong
```

Writes `analysis/phase0_u1_strong.json`.

## Result

| seed | V₀ (confirm) | V* (confirm) | headroom | SE | weak U1 | planted? |
|---|---|---|---|---|---|---|
| 10 | 0.760 | 0.769 | **0.0085** | 0.0013 | 0.0020 | no |
| 11 | 0.736 | 0.745 | **0.0093** | 0.0009 | 0.0096 | no |
| 12 | 0.754 | 0.766 | **0.0123** | 0.0012 | 0.0096 | no |
| 13 | 0.759 | 0.765 | **0.0062** | 0.0011 | 0.0035 | no |
| 14 | 0.733 | 0.754 | **0.0209** | 0.0011 | 0.0129 | no |
| **mean** | | | **0.0114** | | **0.0075** | 0/5 |

The gap is real: every seed is several standard errors above zero. Weak U1 was
low, not invented. Strengthening it moved the mean from 0.0075 to 0.0114.
`QUALITY_WEIGHTS` is **not** the confirmed winner on any seed. The confirmed
`w*` consistently lowers like-weight and hits the share-weight box bound
(10.0 on 4/5 seeds).

## What this decides

The world is flat on top. Available upside is ~0.01 of daily return. Available
downside, from the phase-1 trap, is ~0.30. The ratio is about 1:25. B3 losing
to B1 is therefore not an oracle bug, and the pre-registered MDE of 0.10 in
normalised regret is still incoherent: the denominator is 0.01.

This is the situation the pre-registration already named: *if a calibration
criterion fails, the world is redesigned and the pilot repeated.* C3 failed
because there is almost nothing for B3 to find that B1 would miss, other than
the trap. Recalibration is now an evidenced next step, not a suspicion.
