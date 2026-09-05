# 12 — Fidelity sweep (exploratory, not H3)

**P2 asked for a curve.** This is that curve, labelled exploratory: same five
evaluation seeds (10–14), no locked `N`, and not the registered H3 comparison
(M1 vs B3). The honest bar is still B5.

```bash
python -m rl_opt.run_fidelity
```

Writes `analysis/phase2_fidelity.json` and `analysis/figures/p2_fidelity.png`.

## The dial

Hand-specified, not fitted (Q9). The agent maximises

```
score(w) = φ · V(w) + (1 − φ) · bait_share(w)
```

on a 400-user subset, for free. φ = 1 is the true objective. φ = 0 is the Q8
missing-feature failure: the sim cannot see that bait hurts retention, so it
points at the trap.

M1 searches that score, then **confirms on TrueWorld** (default + sim winner,
100 users each) and ships the better label. B5 is unchanged: eleven delayed-V
labels, no sim.

This is a different M1 from the fitted GP in [`09-phase2.md`](09-phase2.md).
That method already lost to B5. This sweep asks the upper-bound question:
how correct does a simulator have to be before searching it beats buying those
eleven labels?

## Result

| φ | what the sim maximises | mean V | beats B5 |
|---|---|---|---|
| 0.00 | bait share | 0.449 (= B0) | 0/5 |
| 0.25 | mostly bait | 0.449 (= B0) | 0/5 |
| 0.50 | half/half | 0.449 (= B0) | 0/5 |
| 0.75 | mostly V | **0.563** | 3/5 |
| 1.00 | true V | **0.568** | 4/5 |

B0 = 0.449. B5 = 0.563.

**Crossover vs B5 at φ ≈ 0.75.**

Below that, the sim picks bait, the TrueWorld confirm rejects it, and M1 ships
the default. The confirm is doing its job; the sim is adding nothing. At
φ = 0.75 the sim starts handing the confirm a quality-ish config, and M1 ties
B5. A perfect copy of V (noisy, 400 users, 194 candidates) beats eleven labels
by 0.005 and wins 4/5 seeds.

## What this says

A simulator helps only when it is **mostly the objective**. Half-right is the
same as absent: you fall back to default and lose 0.11 of retention to anyone
who just measured V eleven times.

That locates the fitted-GP result. An 11-point ridge/GP is far below φ = 0.75
on this axis — it still walks into bait on some worlds — so losing to B5 is
what the curve predicts, not a surprise.

It also says H3's comparator is still the wrong one. B3's mean V is 0.302.
M1 at every φ, including 0, beats B3, because even shipping default beats
optimising the two-day proxy. The interesting crossing is M1 vs **B5**.

## What this is not

- Not confirmatory H3. Original H3 was vs B3; **amended 2026-09-01** to vs B5.
  `N` is still unlocked. Writeup: [`13-writeup.md`](13-writeup.md).
- Not the registered "realistic fidelity" (logs under `w₀`). This dial is
  bait-blindness, one Q8 axis.
- Not a second fitted-M1 retune.
- Not Q8's full taxonomy (functional form, stale population, supply feedback).
- Not a locked threshold of 0.75. Coarse five-point grid; φ=1 gap vs B5 is 0.005.
