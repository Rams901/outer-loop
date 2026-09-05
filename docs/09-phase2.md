# 09 — M1 vs the delayed-label baseline (exploratory)

**This is not H1.** Same 5 evaluation seeds as phase 1 (10–14), on the
recalibrated world of [`11-recalibration.md`](11-recalibration.md). Stage-2
`N` stays unlocked.

```bash
python -m rl_opt.run_m1_dev   # choose one variant on seeds 20–24
python -m rl_opt.run_m1       # frozen variant vs B0/B1/B3/B4/B5 on 10–14
```

Writes `analysis/m1_retune_dev.json`, `analysis/phase2_m1_exploratory.json`,
and `analysis/figures/p2_m1_committed_v.png`.

## Why B5 exists

B0–B4 never see `V`. Comparing M1 to them only shows that delayed labels help,
which we already knew. **B5** spends the identical 24,000 user-day budget on
full-horizon labels and ships the best config it actually measured. That is
the honest bar: does fitting a model beat buying more measurements?

After the first (unfair) M1 run, both methods were given the **same 11-label
design**: default, the engagement trap, then nine random draws. Same user
slices, same observations. B5 selects the best measured row. M1 may
extrapolate. `QUALITY_WEIGHTS` remains withheld from both.

## The one allowed retune

The first M1 burned three labels' worth of budget on a 340-user confirm and
fit ridge on eight points. That was a weak version of the method. One retune
was allowed, selected only on development seeds **20–24**, then frozen.

| Variant on 20–24 | mean V |
|---|---|
| B5 (no model) | **0.563** |
| conservative GP (LCB β=1) | 0.562 |
| conservative GP (LCB β=0.5) | 0.561 |
| ridge, 11 labels, no confirm | 0.562 |
| GP mean (no penalty) | 0.541 |

Unconstrained GP collapsed on seed 21 (0.459). The selected variant was
**GP + lower-confidence-bound**, Δ vs B5 = −0.001 on development. That is the
method evaluated below.

## Evaluation result (seeds 10–14)

| Method | sees V? | mean V | mean R | Walk-backs |
|---|---|---|---|---|
| **B5** | yes | **0.563** | **0.10** | 0/5 |
| **M1** | yes | 0.552 | 0.19 | 0/5 |
| B0 | no | 0.449 | 1.00 | 0/5 |
| B1 | no | 0.385 | 1.53 | 5/5 |
| B4 | no | 0.364 | 1.69 | 5/5 |
| B3 | no | 0.302 | 2.22 | 5/5 |

**M1 − B5 = −0.011, and M1 wins 2 of 5 seeds.**
Paired: −0.001, **−0.067**, +0.010, +0.005, −0.004.

The retune helped: the previous M1 was 0.537 and lost 5/5. It did not reverse
the sign. Mean V of B5 also dropped slightly (0.568 → 0.563) because the
shared design now includes the trap, which B5 will never ship but which
occupies one of the eleven slots.

## What this says

**The surrogate still does not pay for itself.** After matching the
information channel, dropping the wasteful confirm, and picking the most
conservative model on held-out worlds, interpolating from 11 delayed-V labels
does not beat selecting the best of those 11.

The failure mode is visible on seed 11. B5 ships a measured low-like /
high-share config (V=0.552, bait 1%). M1 extrapolates to like-weight 3.94
(V=0.485, bait 11%) — the classic surrogate-exploitation pattern. On the
other four seeds M1 stays near B5's pick and the gap is noise.

**The separation that survives is still between metrics, not methods.**
Everyone who optimises the 2-day proxy lands below the default. Everyone who
pays for delayed `V` lands ~0.10 above it. Within the second group, breadth of
honest measurement beats a fitted model.

The headline remains **"measure the right thing," not "simulate."**

## Oracle validity

`u1_oracle` now includes an ES refinement on true V. `B5 > V*` is false on
all five evaluation seeds. Regret is defined.

## What this is not

- Not confirmatory H1. Original H1 named B3; that comparison is uninteresting
  (B3 cannot see `V`). **Amended 2026-09-01:** confirmatory H1 is M1 vs B5.
  See [`../PREREGISTRATION.md`](../PREREGISTRATION.md) and
  [`13-writeup.md`](13-writeup.md).
- Not a fidelity sweep of the *fitted* GP. A hand-specified dial is in
  [`12-fidelity.md`](12-fidelity.md): M1 crosses B5 at φ≈0.75.
- Not a second retune. The protocol allowed one, on development seeds, and
  this was it.
- Not M2 / RL. Not C4. P3 is deferred until confirmatory M1 beats B5.
