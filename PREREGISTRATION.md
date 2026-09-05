# Pre-registration

**Status: Stage 1 amended 2026-09-01. Stage 2 locked 2026-09-01. Confirmatory H1 complete.**
See `docs/14-stage2.md`, `docs/17-h1.md`, and `analysis/confirmatory_h1.json`.

---

## Why this document exists

We are building the benchmark *and* competing in it. That is a structural
conflict of interest, and good intentions are not a remedy for it. Without
committing to the analysis in advance, there is nothing stopping us from
running the experiment, noticing which comparison happens to favour our method,
and reporting that one as though it had been the plan.

This file removes that freedom. Anything not specified here is exploratory and
must be labelled as such in the writeup.

## Two stages

Some parameters cannot honestly be fixed before a pilot — we do not yet know the
variance of the outcome, so we cannot compute a sample size. Pretending
otherwise would be its own form of dishonesty.

**Stage 1 (locked, then amended 2026-09-01).** Hypotheses, primary metric,
primary comparison, analysis plan, calibration criteria, falsification
conditions, tuning policy. Original text is retained; the amendment supersedes
H1, H3, H5, the primary test, and the falsification comparator.

**Stage 2 (locked 2026-09-01).** `N = 50` from the exploratory paired (M1 − B5)
regret sd (0.243), Wilcoxon-inflated, clipped to the floor/ceiling. Confirmatory
seeds 30–79. Budget 24,000 user-days. `L = 11` delayed-V labels (not logs under
`w₀`; see deviation log). Frozen M1 = GP LCB β=1. World parameters: current
`WorldParams` + satisfaction knobs in `docs/14-stage2.md`.

Pilot seeds are discarded and never reused in the confirmatory run.

---

## Definitions

Let `V(w)` be the true long-horizon objective of config `w` under `TrueWorld`,
estimated to high precision with unlimited simulated samples. This is legitimate
because the evaluator is not budget-constrained; only the *agent* is.

- `w*` — the safety-feasible optimum, found by brute force (entrant U1)
- `w₀` — the default config
- `ŵ` — the config a method commits to when its budget is exhausted

**Primary metric — normalised regret:**

```
R = ( V(w*) − V(ŵ) ) / ( V(w*) − V(w₀) )
```

- `R = 0` — found the optimum
- `R = 1` — no better than shipping the default and spending nothing
- `R > 1` — actively worse than doing nothing

**Unsafe commits.** If `ŵ` violates a safety floor, `R` is recorded as **2.0**.
This is a fixed penalty chosen in advance, not fitted to results. The
invalid-commit rate is also reported separately, so this cannot be hidden inside
an average.

**Realistic fidelity.** `SimWorld` fitted from `L` user-days of logs generated
under `w₀`, where `L` is fixed in stage 2. This is the operational definition
used in H1 — not a judgement call made after seeing the fidelity sweep.

---

## Hypotheses

Original text below. For confirmatory analysis, the 2026-09-01 amendment
supersedes H1, H3, H5, the primary test, and the falsification comparator.

**H1 — primary.** At realistic fidelity, simulator-guided search (M1) achieves
lower normalised regret than Bayesian optimisation run directly against
`TrueWorld` (B3), at equal user-day budget.

B3 is deliberately the comparator rather than the human grid B1. Beating a
four-point grid search would be unimpressive and easy; beating well-tuned
Bayesian optimisation is the claim that would actually matter.

**H2 — secondary.** M1 achieves lower normalised regret than the human-style
grid (B1). Expected. If this fails, something is wrong with the setup rather
than interesting.

**H3 — secondary.** Normalised regret for M1 increases monotonically as
surrogate fidelity decreases, and there exists a fidelity level below which M1's
regret exceeds B3's.

**H4 — secondary.** M1 has a lower walk-back rate than B1.

**H5 — secondary.** M1's advantage over B3 increases with the reporting delay on
long-horizon metrics.

**M2** (the sequential experiment-design agent) enters the same comparisons as
M1 once phase 3 exists. Its hypotheses are not registered here because it is not
yet designed; they will be added as a stage 3 amendment before it runs.

---

## Analysis plan

**Design.** Paired. Each seed instantiates one ground truth; every entrant runs
against the identical set of seeds.

**Primary test.** Wilcoxon signed-rank on the paired differences in normalised
regret (M1 − B3), two-sided, α = 0.05. Non-parametric because regret
distributions are expected to be skewed and bounded below.

**Effect size.** Median paired difference with a 95% bootstrap confidence
interval, 10,000 resamples. The confidence interval is the headline number, not
the p-value.

**Multiple comparisons.** H2–H5 are corrected with Holm–Bonferroni as a family.
H1 stands alone as the single pre-registered primary comparison.

**Everything else is exploratory** and will be labelled as such, including all
ablations in `05-protocol.md` beyond H3 and H5.

### Sample size

- Minimum detectable effect, pre-specified: **median paired difference of 0.10**
  in normalised regret. Smaller differences are not interesting enough to chase.
- Pilot: 5 ground-truth seeds during phase 0/1, used only to estimate variance.
- `N` from a power analysis at 80% power, α = 0.05, using the pilot variance.
  Floor 20, ceiling 200.
- `N` is fixed in stage 2 and does not change afterwards.

### Tuning policy

The easiest way to fake a win is to tune your own method hard and the baseline
carelessly.

- All hyperparameter tuning happens on a **development seed set, disjoint from
  the confirmatory seeds**.
- Every entrant receives an **equal tuning budget**, measured in number of
  configurations evaluated. The number is fixed in stage 2.
- B3 and B4 receive tuning effort at least equal to M1 and M2. If in doubt, the
  baselines get more.

### Stopping rules

- Run the full `N`. No interim inspection of the primary comparison.
- No adding seeds after seeing results.
- Crashed runs are re-run with the identical seed and logged in the deviation
  log. A run is never silently dropped.

---

## Calibration criteria

These gate the confirmatory run. Each is checked during phase 0/1 on pilot
seeds. **If a criterion fails, the world is redesigned and the pilot repeated —
we do not proceed with a broken benchmark and explain it away afterwards.**

**C1 — the trap exists.** There is a config `w_bait` whose short-horizon proxy
exceeds that of `w₀` by at least 5%, while `V(w_bait) < V(w₀)`. Without this
there is no Goodhart phenomenon and nothing to study.

**C2 — the problem is not trivial.** Random search (B2) does not reach `R < 0.2`
within the budget. If random search solves it, the config space is too small or
too smooth.

**C3 — the incumbent is beatable.** B3 achieves meaningfully lower regret than
B1. If a four-point grid matches Bayesian optimisation, the space is too simple
to distinguish any methods at all.

**C4 — heterogeneity is real.** The gap between the best global config and the
best per-segment config is at least 20% of `V(w*) − V(w₀)`. This confirms the
population is diverse enough that a global config is a genuine compromise, and
it establishes in advance whether the contextual extension is worth building.

---

## Falsification

Stated now, while there is no incentive to be slippery about it.

**The project is abandoned or redesigned if:** phase 0 cannot satisfy C1 after
two redesign attempts. A world without a trap cannot answer the question.

**The thesis is reported as refuted if:** M1 fails to beat B3 at *every* fidelity
level, including perfect fidelity where `SimWorld` is `TrueWorld`. That would
mean the surrogate approach adds nothing in this setting regardless of how good
the surrogate is, which is a clean and useful negative result.

**The headline becomes the crossover rather than the win if:** M1 beats B3 only
above a fidelity level that is not reachable by fitting from a realistic volume
of logs. That is the most likely outcome and it is a fine one — the deliverable
becomes the threshold itself.

A negative result is published with the same prominence as a positive one. This
sentence exists so that it is awkward to do otherwise later.

---

## Locked in stage 1 (original)

Hypotheses H1–H5 · primary metric and its normalisation · unsafe-commit penalty
of 2.0 · primary comparison M1 vs B3 · Wilcoxon signed-rank with bootstrap CI ·
Holm–Bonferroni for the secondary family · MDE of 0.10 · power 80% at α = 0.05 ·
seed floor 20 and ceiling 200 · equal tuning budgets with baselines favoured ·
calibration criteria C1–C4 · falsification conditions

---

## Stage 1 amendment — 2026-09-01

Made **before** any confirmatory run and **before** locking stage 2. Exploratory
results on seeds 10–14 are not this amendment's evidence base for a win; they
are why the original comparison is unfalsifiable.

**Why.** B3 maximises a two-day proxy and never observes `V`. After
recalibration, shipping the default beats B3 on every seed. H1 as originally
written (M1 vs B3) is therefore true for an uninteresting reason: *any* method
that does not Goodhart the proxy beats B3. That does not test whether a
simulator reduces live experiments.

**B5** spends the same user-day budget on delayed `V` and ships the best
measured config. M1 vs B5 is the thesis: does fitting or querying a simulator
beat buying more honest measurements?

### Hypotheses (supersede the originals for confirmatory analysis)

**H1 — primary (amended).** At realistic fidelity, M1 achieves lower
normalised regret than B5, at equal user-day budget.

**H1-original — secondary.** M1 achieves lower normalised regret than B3.
Expected. Already true in exploratory data because B3 Goodharts.

**H2 — secondary.** Unchanged: M1 vs B1.

**H3 — secondary (amended).** There exists a fidelity level below which M1's
committed `V` is below B5's. (Exploratory bait-blindness dial: φ ≈ 0.75. That
number is not locked.)

**H4 — secondary.** Unchanged: M1 walk-back rate vs B1.

**H5 — secondary (amended).** M1's advantage over **B5** increases with
reporting delay on long-horizon metrics. Original H5 vs B3 is retired.

**M2.** Not designed. **Deferred** until confirmatory H1 vs B5 is positive in
some fidelity region. Exploratory evidence is negative. No stage 3 amendment
until then.

### Analysis plan (amended)

**Primary test.** Wilcoxon signed-rank on paired differences in normalised
regret **(M1 − B5)**, two-sided, α = 0.05. Median paired difference with 95%
bootstrap CI, 10,000 resamples, remains the headline interval.

Report **mean committed V** alongside R. R is the test statistic; V is what a
reader should look at. This does not replace R.

**Multiple comparisons.** H1-amended stands alone. The secondary family is
H1-original, H2, H3-amended, H4, H5-amended (Holm–Bonferroni).

### Calibration (amended)

C1 and C2 remain gates. **C3 failed** after redesign (B3 does not beat B1) and
is **not** a gate for the confirmatory run. Replacement gate, already used in
the exploratory pilot: `V* − V₀ ≥ 0.05`, C1, C2, and every proxy-maximising
method loses to B0. C4 remains unrun and still gates only the contextual
extension.

### Falsification (amended)

**The thesis is reported as refuted if:** M1 fails to beat **B5** at every
fidelity level, including φ = 1 where the sim maximises true `V`. Exploratory
data already refute a *fitted* 11-point GP vs B5; they do not yet refute a
near-oracle sim (φ = 1 won 4/5 seeds, gap 0.005).

**The headline is the crossover if:** M1 beats B5 only above a fidelity that
fitting from a realistic log volume cannot reach. Exploratory bait-blindness
puts that threshold near φ = 0.75.

### What this amendment does not do

- Does not lock stage 2 (`N`, `L`, world, budget, confirmatory seeds).
- Does not reuse seeds 10–14 as confirmatory. They remain exploratory forever.
- Does not change the MDE (0.10 in R), α, power, or seed floor/ceiling.
- Does not authorise a second M1 retune.

## Locked stage 2

`N = 50` · confirmatory seeds 30–79 · budget 24,000 user-days · `L = 11`
delayed-V labels (100 users, 21-day horizon) · frozen M1 `m1_gp_lcb_one` ·
world = post-recalibration `WorldParams` / satisfaction knobs in
`docs/14-stage2.md`

## Deviation log

Every departure from this document is recorded here with a date and a reason,
including deviations that seem harmless. An empty log at the end of the project
would be more suspicious than a populated one.

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-09-01 | C4 | Evaluated; **failed** (mean gap/headroom = 0.004 vs 0.20) | Primary-topic segments. Gates contextual extension only. H1 proceeds. `docs/15-c4.md` |
| 2026-08-27 | C3 | Failed on the 5-seed pilot (mean R_B3 > R_B1) | Recorded as a result, not a protocol change. P1 exit still met via B4 |
| 2026-08-27 | Sample size / H1 | Exploratory M1 vs B0/B1/B3/B4 on the **same 5 pilot seeds** | Not confirmatory. Pilot seeds are not discarded here; stage-2 N stays unlocked. Labelled exploratory in `docs/09-phase2.md` |
| 2026-08-27 | Realistic fidelity | M1 fits `SimWorld` from charged full-horizon labels at several `w`, not from logs under `w₀` only | Tests delayed-return labels as the information channel. Registered H1 still requires the `w₀`-log definition |
| 2026-08-27 | U1 | Diagnostic `u1_strong` (394 oracle calls, held-out confirm) run on the same 5 pilot seeds | Not a protocol change. Shows `V*−V₀` is real (~0.011) and still too small for the registered MDE |
| 2026-08-27 | World / C3 | Satisfaction and default weights redesigned (`docs/11-recalibration.md`) | C3 failed because the world was flat on top. Protocol says redesign and repeat the pilot. Stage 2 still unlocked |
| 2026-08-28 | P1 exit | After redesign, B4 no longer beats B1 | Logged, not silently rewritten. Replacement gate: headroom ≥ 0.05, C1, C2, proxy methods lose to B0 |
| 2026-08-28 | Entrants | Added **B5**: same budget, same access to delayed `V`, no surrogate | B0–B4 cannot see `V`, so M1 beating them measured the information channel, not the method. B5 is the honest bar |
| 2026-08-28 | Entrants | Removed `QUALITY_WEIGHTS` from M1 and B5 candidate pools | Oracle-derived from the satisfaction model; seeding an entrant with it is an answer key. U1 retains it |
| 2026-08-28 | U1 | Added ES refinement on true `V` | An entrant (B5) beat the old `V*`, making regret negative and meaningless |
| 2026-08-28 | H1 comparator | Exploratory result suggests **B3 is the wrong comparator** | B3 cannot observe `V`. H1 as registered is winnable for an uninteresting reason. Flagged before any confirmatory run; **amended 2026-09-01** |
| 2026-08-28 | M1 / tuning | One retune, selected on development seeds 20–24, frozen, then evaluated on 10–14 | Protocol allows one development-set tune. Shared 11-label design; confirm dropped; conservative GP (LCB β=1) selected. M1 still loses to B5 (ΔV = −0.011, 2/5 seeds) |
| 2026-09-01 | H3 / P2 | Exploratory fidelity sweep on seeds 10–14 (`docs/12-fidelity.md`) | Not confirmatory. Dial is bait-blindness, not logs under `w₀`. Comparator reported vs B5. Crossover φ≈0.75 |
| 2026-09-01 | H3 / Q8 | Exploratory four-axis sweep on seeds 10–14 (`docs/16-q8.md`) | Bait remains the harsh axis. Stale/rare error is mild. Not confirmatory H3. |
| 2026-09-01 | Stage 1 | **Amendment:** primary H1 = M1 vs B5; H3 crossover vs B5; H5 vs B5; C3 not a confirmatory gate; M2/P3 deferred | Made before any confirmatory run and before locking `N`. Original H1 vs B3 kept as a labelled secondary. Writeup: `docs/13-writeup.md` |
| 2026-09-01 | Stage 2 | **Locked:** N=50, seeds 30–79, L=11 delayed-V labels, frozen GP LCB β=1 | Pilot ΔR sd=0.243 on seeds 10–14 (Wilcoxon-inflated). Those seeds unused as confirmatory. `docs/14-stage2.md` |
| 2026-09-01 | Realistic fidelity | Confirmatory H1 uses the 11-label delayed-V design, not logs under `w₀` | The allowed retune already froze that method. A new w₀-log M1 would be a second method. |
| 2026-09-01 | H1 | Confirmatory complete (`docs/17-h1.md`) | Wilcoxon p=0.013, median ΔR=−0.025, CI [−0.038, 0]. Mean V M1 0.562 vs B5 0.560. MDE 0.10 not met. M2 still deferred. |
