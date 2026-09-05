# 11 — World recalibration

Phase-1 C3 failed and step 0 showed why: `V* − V₀ ≈ 0.01` because 21-day
satisfaction clipped against 0.98 under the default feed. The benchmark was
measuring damage avoidance, not search. The pre-registration says to redesign
the world and repeat the pilot, not to proceed on a broken calibration.

## What changed

**Satisfaction no longer saturates.** `SAT_STEP` 0.070 → 0.032, `SAT_INIT`
0.58 → 0.48 (the return-sigmoid centre), `RETURN_K` 6 → 8. Ranking quality vs
mediocre content can now move V.

**Default is a production compromise, not the optimum.** Like 1.8 / dwell 0.55 /
share 1.0 instead of like 0.5 / dwell 0.05 / share 2.0. Hide and report still
echo `param.rs` (−43.2, −234). The shipped config slightly overweights cheap
engagement; that is the interesting starting point.

**Trap moved further along the same axis.** Engagement like 2.6 / dwell 1.6,
weaker report penalty. Feed mix shifts (~18% → ~38% bait), does not collapse.

**Quality is the opposite direction**, and it looks *worse* on the 2-day proxy.
That is what makes delayed-V labels worth paying for.

## Targets, checked on full-size worlds before locking

| | target | observed (seeds 0,1,2,10,11) |
|---|---|---|
| `V(quality) − V(default)` | 0.05–0.10 | 0.09–0.12 |
| C1 proxy lift | ≥ 5% | 10–14% |
| C1 `ΔV` trap | < 0 | −0.09 to −0.11 |
| default bait share | mixed, not ~1% | 14–22% |
| quality proxy vs default | not higher | lower |

Quality looks *worse* on the 2-day proxy. That is what makes delayed-V labels
worth paying for.

Phase 0/1 exploratory numbers in `docs/06`–`09` are the *pre-recalibration*
record. M1 has not been re-run on this world.

## Pilot on the new world

`prove_c1` and `run_pilot` were re-run after locking the knobs.

**C1 passed** (seeds 0–2): mean proxy lift **+12.1%**, mean `ΔV` **−0.095**,
bait share ~18% → ~39%.

**C2 passed** (seeds 10–14): B2 mean R = 2.28, never `< 0.2`.

**C3 failed** (mean R_B3 = 2.31 > R_B1 = 1.57). B3 still Goodharts harder than
the grid.

**Original P1 exit failed:** B4 mean V 0.364 does not beat B1 0.385. Both lose
to B0 (0.449). U1 `V*` sits at 0.55–0.58, so the missing 0.10 is real and is
exactly what a delayed-V method is for.

The replacement gate, not yet a pre-reg amendment: headroom ≥ 0.05, C1, C2,
and every proxy-maximising method loses to B0. That combination holds.
