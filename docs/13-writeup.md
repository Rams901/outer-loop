# 13 — Interim writeup (P4)

Exploratory evidence plus a stage-1 protocol amendment. **Not a confirmatory
report.** Stage 2 (`N`, world lock, confirmatory seeds) is still unlocked.
Seeds 10–14 are exploratory forever.

Someone else should be able to clone this repo and regenerate the headline
figure. That is the P4 exit.

```bash
pip install -e '.[dev]'
pytest
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.prove_c1
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.run_pilot
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.run_m1_dev
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.run_m1
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.run_fidelity
```

Headline figure: `analysis/figures/p2_fidelity.png`, from
`python -m rl_opt.run_fidelity`. Numbers: `analysis/phase2_fidelity.json`,
`analysis/phase2_m1_exploratory.json`, `analysis/phase1_pilot.json`.

`OMP_NUM_THREADS=1` avoids BLAS oversubscription on long jobs. Pre-recalibration
tables in `docs/06`–`08` are a different world; do not mix them with `09`/`12`.

---

## The question

Can offline simulation reduce live A/B tests for ranking **weights**, and when
does the sim stop being trustworthy? Outer loop = config tuner. Inner ranker is
fixed: `score = w · p̂`. Objective = long-horizon retention `V` under hard
safety floors.

The honest version of the claim is not "replace A/B testing." It is: a
simulator can change what you spend live traffic on, if it is right enough
that searching it beats buying more measurements of `V`.

---

## Claims (exploratory, seeds 10–14)

These are qualitative claims that survive the n=5, coarse-grid caveats. Means
are rounded; exact values are in the JSON.

**1. Proxy search Goodharts.** Methods that maximise a two-day proxy (B1–B4)
all land *below* the default. B0 mean V **0.449**. B1 **0.385**. B4 **0.364**.
B3 **0.302**. Walk-back 5/5 for every proxy method. C1 holds: proxy lift with
ΔV negative and bait roughly doubling.

**2. Measuring delayed V recovers about 0.10 of retention.** B5 spends the
same 24,000 user-day budget on eleven full-horizon labels and ships the best
measured config. Mean V **0.563**. That gap is the information channel, not a
search algorithm.

**3. A fitted simulator does not beat those eleven labels.** After one allowed
retune (dev seeds 20–24), frozen GP LCB β=1 vs B5 on seeds 10–14: M1 **0.552**
vs B5 **0.563**, M1 wins **2/5**, paired Δ ≈ **−0.011**. On seed 11 M1
extrapolates to like-weight ≈3.94 and 11% bait (surrogate exploitation). The
other four seeds are noise around B5's pick.

**4. A half-right sim is the same as no sim.** On the bait-blindness dial,
`score = φ·V + (1−φ)·bait_share`, M1 searches the sim for free then confirms
two TrueWorld labels. For φ ≤ 0.5, M1 ships the default (0.449) on 5/5 seeds.
The confirm rejects the trap; the sim added nothing.

**5. The sim starts helping only once it is mostly the objective.** φ = 0.75:
mean V **0.563**, beats B5 on **3/5**. φ = 1: **0.568**, **4/5**, gap **0.005**.
Coarse-grid crossover vs B5 ≈ **0.75**. That number is not a precise threshold.

---

## Non-claims

- **Not confirmatory H1 or H3.** No locked `N`. Same five eval seeds as the
  pilot. Pre-reg said discard pilots.
- **Not "fitted sim reduces A/B tests."** Fitted M1 lost to B5. That is the
  result, not a near-miss to be tuned away.
- **Not H1 vs B3 as a thesis test.** B3 never sees `V`. Shipping default beats
  B3. Any method that does not Goodhart the proxy "wins" H1-original. The
  2026-09-01 amendment makes primary H1 = M1 vs **B5**.
- **Not a locked fidelity threshold of 0.75.** Five-point grid, one Q8 axis
  (bait-blindness), interpolated `crossover_phi_vs_b5` ≈ 0.749 in JSON. The
  robust statement is "half-right fails, mostly-right ties, oracle-sim is a
  0.005 gap."
- **Not two interchangeable M1s.** Fitted GP (11 delayed-V labels, no confirm)
  is `docs/09`. Dialled `FidelitySim` + two-label confirm is `docs/12`. They
  answer different questions.
- **Not M2 / RL / C4.** Not started. Deferred (see below).
- **Not a mixture of pre- and post-recalibration numbers.** `docs/11` redesigned
  satisfaction. `06`–`08` are the old world.

---

## Protocol (what changed, what did not)

Original stage 1 named B3 as the H1 comparator because B3 was supposed to be
well-tuned Bayesian optimisation on the true expensive objective. After
implementation, short experiments set `Receipt.value=None`; B3 maximises the
proxy. Recalibration then made the proxy a trap, so C3 failed (B3 worse than
B1). The replacement calibration gate, already used in the exploratory pilot:
`V* − V₀ ≥ 0.05`, C1, C2, every proxy-maximising method loses to B0.

**Amendment 2026-09-01** (in `PREREGISTRATION.md`, before any confirmatory run,
before locking stage 2):

- Primary H1: M1 vs **B5**, Wilcoxon on paired (M1 − B5) in R.
- H3: crossover vs B5, not B3.
- H5: vs B5; original vs B3 retired.
- H1-original (vs B3) kept as a labelled secondary.
- C3 is not a confirmatory gate.
- M2 / P3 deferred until confirmatory H1 vs B5 is positive in some fidelity
  region.
- Stage 2 still not locked. Seeds 10–14 remain exploratory.

B5 and fitted M1 share `delayed_label_design()` and the same harness RNG seed,
so they see the same eleven observations. Only the decision rule differs.
`QUALITY_WEIGHTS` is withheld from both.

---

## Limitations

**n = 5.** Variance of V is large enough that a 0.005 gap (φ = 1 vs B5) is
fragile. Do not lock confirmatory `N` from this pilot.

**One Q8 axis.** The dial is "sim cannot see that bait hurts retention."
Functional-form error, stale population, and supply feedback are untested.
A crossover on bait-blindness is not a crossover on "realistic logs under `w₀`"
(the registered definition of realistic fidelity, still unlocked in stage 2).

**Two confirmation labels vs eleven measurements.** Dialled M1 spends most of
its budget on *free* sim queries and two TrueWorld confirms. B5 spends the
whole budget on labels. That is the comparison the thesis needs, but it is not
an equal-information comparison.

**One M1 retune, spent.** Protocol allowed one development-set tune (seeds
20–24). A second would be fishing.

**U1 is a diagnostic, not a competitor.** Oracle ES on true V exists so that
regret is defined (`B5 > V*` is false on these seeds). It is not an entrant.

---

## What we will not do next

- Lock stage 2 `N` from n = 5.
- Start P3 / M2 / PPO unless a *confirmatory* M1 beats B5 somewhere on the
  fidelity curve. Exploratory evidence is negative for the fitted method and
  only barely positive for an oracle sim.
- Run a second fitted-M1 retune.
- Quote φ ≈ 0.75 as a confirmatory H3 result.
- Reuse seeds 10–14 as confirmatory.

---

## What remains if the project continues

A confirmatory run needs a new seed set, a locked `N` from a power calculation
that is not this n = 5, and the amended H1 (M1 vs B5). Other Q8 axes, H5
(reporting delay vs B5), and C4 (contextual configs) are still unrun.

Until confirmatory H1 vs B5 is positive, **P3 is skipped.** Adding an RL
agent that chooses the next experiment does not repair a simulator that is
not the objective.

The headline that is already supported:

> Measure the right thing. A surrogate fitted from a realistic number of
> delayed-V labels does not beat selecting the best of those labels. A
> simulator that is only half the objective is discarded by the confirm
> and you ship the default. Simulation starts to pay only once it is
> mostly `V` — and even then the gap over honest measurement is small.
