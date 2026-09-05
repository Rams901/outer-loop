# 08 — Phase 1 report

**Exit criterion:** harness + B0–B4 + U1, regret across seeds, B3 *or* B4
clearly beating B1.

> The table in “Result” is the *pre-recalibration* pilot. After
> [`11-recalibration.md`](11-recalibration.md) the same command produces the
> numbers in “Post-recalibration.” The original P1 exit (B4 beats B1) does
> **not** hold on the new world: every proxy-maximising method loses to B0.

```bash
python -m rl_opt.run_pilot
```

Pilot spec (not confirmatory, not stage-2 locked): 24,000 user-days, 800 users ×
2 days per look, 5 world seeds (10–14). Agents see **only the 2-day proxy**.
`V` is scored afterwards by the oracle, uncharged.

## Q15, pinned for this pilot

One user-day = one user, under a non-default config, for one simulated day.
Treatment is charged. The oracle is not. Short experiments (`n_days=2`) return
`Receipt.value = None`, so a method cannot “accidentally” optimise retention.

## Result

| Method | What it does | mean V | mean R | Walk-backs |
|---|---|---|---|---|
| **B0** | ship default, spend nothing | **0.749** | 1.00 (floor) | 0/5 |
| **B1** | 4-point like-weight grid | 0.583 | 32.0 | 5/5 |
| **B2** | random search on proxy | 0.416 | 68.7 | 5/5 |
| **B3** | GP / EI on proxy | 0.452 | 61.4 | 5/5 |
| **B4** | log-weight evolution | **0.739** | **3.07** | 5/5 (tiny) |

`R = (V* − V̂) / (V* − V₀)`. B0 is identically 1 when it ships the default.
`R > 1` means worse than doing nothing. The huge B1–B3 numbers are that
formula with a **tiny** `V* − V₀` (~0.002–0.013): default is already near the
U1 best, so any trap-walk looks enormous. Read **mean V** for the human-scale
picture.

**C2 passed.** B2 never reached `R < 0.2`. Random search on the proxy does not
solve the problem — it walks into bait (population bait share 0.4–0.7).

**C3 failed as registered.** B3 did *not* beat B1. Bayesian optimisation on the
lying metric Goodharts as hard as the grid, sometimes harder. That is not a
broken optimiser; it is the observation model doing what we built it to do.

**P1 exit still holds on that world:** B4 clearly beats B1 (mean V 0.739 vs 0.583). Conservative
local search stays near default; aggressive proxy search does not.

**C4 not run.** Per-segment vs global optimum is deferred. P1 did not need it.

## What this says

The expensive part of ranking-policy search is not “find a clever `w`.” Default
is already close to `w*` on true retention. The expensive part is **not
shipping a config that looks good in two days**. B1/B2/B3 all raise like+dwell
and quietly lose ~0.15–0.33 of daily return. B4 barely moves `w` and barely
loses V.

Phase 2 (surrogate) has a clean job: a `SimWorld` that is *wrong in the same
way as the proxy* will not help. One that can see the delayed retention
structure might. C3 failing on B3 vs B1 is a reason to want that, not a reason
to skip it. Exploratory M1 is in [`09-phase2.md`](09-phase2.md); it is not H1.

## Post-recalibration (current world)

Same 5 seeds (10–14), same budget, same “max the 2-day proxy then commit”
entrants. `V* − V₀` is now 0.09–0.14, so `R` is a usable number rather than a
ratio of noise.

| Method | mean V | mean R | Walk-backs |
|---|---|---|---|
| **B0** | **0.449** | 1.00 | 0/5 |
| B1 | 0.385 | 1.57 | 5/5 |
| B4 | 0.364 | 1.75 | 5/5 |
| B2 | 0.304 | 2.28 | 5/5 |
| B3 | 0.303 | 2.31 | 5/5 |

**C2 passed.** B2 never reached `R < 0.2`.

**C3 failed again, and now we know why.** B3 is a better optimiser of a lying
metric than B1 is. The original P1 exit (B4 beats B1) also fails: B4 still
wanders toward the proxy and takes a smaller version of the same damage.

B0 is the bar. A method that cannot see delayed V should not beat shipping
the default. That is the needle M1 is supposed to move.

## Harness

- `Budget` raises rather than overspending.
- `Harness.experiment` is the only TrueWorld path for entrants.
- Tests: short receipts hide `V`; over-budget charges fail closed.
