# 05 — Protocol

## Currency

Everything is denominated in **user-days of exposure**: one user, experiencing a
non-default config, for one simulated day.

Rules, to be pinned down precisely before any numbers are reported (Q15):

- Treatment arms are charged. Holdout is not.
- Ramp periods are charged at their actual size.
- Polling for a delayed metric after exposure has ended is free — the harm was
  already done and paid for.
- Querying `SimWorld` is always free.

Every method gets the same total. That is the only reason cross-method
comparison means anything.

---

## Entrants

All baselines must be implemented and working before any claim is made about any
method. The baselines are not a formality — B1 is the incumbent and the thing
most likely to embarrass us.

| ID | Strategy | Why it is here |
|---|---|---|
| **B0** | Ship the default config, spend nothing | Sanity floor. Any method that loses to this is broken. |
| **B1** | Human-style grid — pick one axis, four values, equal-split A/B, take the winner | The incumbent. Deliberately mirrors the bidirectional-boost episode. |
| **B2** | Random search on the **proxy** | Embarrassingly strong in high dimensions; here it Goodharts like B3/B4. |
| **B3** | GP/EI on the **two-day proxy** (clever Goodhart control) | Originally described as BO on `TrueWorld`. Short experiments do not observe `V`, so this is the method that walks into the trap. |
| **B4** | Log-weight ES on the proxy | Same information channel as B3, different optimiser. |
| **B5** | Eleven full-horizon `V` labels, ship the best measured | The honest bar after the 2026-09-01 amendment. Same budget as M1, same access to delayed `V`, no surrogate. |
| **M1** | Fit or query `SimWorld`, optionally confirm top-k on `TrueWorld` | The straightforward version of the project's thesis. |
| **M2** | Sequential experiment-design agent | The RL contribution. **Deferred** until confirmatory M1 beats B5 somewhere. |
| **U1** | Optimise directly against `TrueWorld` with unlimited budget | Not an entrant — computes the optimum for regret. |

B5 is the honest bar. B3/B4 never see `V`; beating them only shows that the
proxy is not the objective. If M1 cannot beat buying more delayed-`V`
measurements, the simulator adds nothing, and that is the headline.

---

## Metrics

**Primary — true value of the committed config.** Not the best ever observed.
The agent must commit under uncertainty, and is scored on what it shipped.

**Regret** against the brute-forced optimum from U1.

**Budget-to-threshold** — user-days required to reach a fixed fraction of
optimal. Captures efficiency rather than just final quality, and is the number
that would actually matter to a team.

**Walk-back rate** — fraction of runs whose committed config would later be
reverted under the true objective. A method that finds slightly worse configs
but never ships a reversal may be the better method, and this is the metric that
shows it.

**Safety violations during search** — cumulative constraint breaches incurred
while searching. Reaching the optimum by torturing users is not a win.

**Sim-to-real correlation** (M1, M2 only) — rank correlation between surrogate
predictions and true values over evaluated configs. Diagnostic rather than a
score.

---

## Statistical discipline

Two independent sources of randomness — the instantiation of the ground truth,
and stochastic rollouts within it — and both need replication.

- Seeds: target ≥20 ground-truth instantiations, but set the final number from a
  pilot variance estimate rather than picking a round figure.
- Report full distributions, not means alone. Paired comparisons across seeds
  where methods share a ground truth.
- Fix seeds across methods so every entrant faces the same worlds.
- No tuning any method on the test seeds. Hyperparameters get set on a
  development set of seeds that never appear in reported results.

### Pre-registration

We are building the benchmark and competing in it. That is a conflict of
interest and needs a structural remedy, not good intentions.

Before the first full run, commit a file specifying the primary metric, seed
count, success thresholds, and the exact comparison being made. Git history
timestamps it. If we later change the analysis, the change is visible and
explained rather than silent.

---

## Ablations

**Fidelity sweep.** The headline figure: regret against surrogate fidelity,
locating the crossover where simulation stops helping. Run separately for each
kind of wrongness from Q8, since collapsing them onto one axis probably hides
the interesting structure.

**Reporting delay.** Sweep the lag on long-horizon metrics. Expect
simulator-guided methods to gain as delay grows, because that is precisely when
direct experimentation is most crippled. If that does not show up, something is
wrong with the setup.

**Heterogeneity.** Sweep population diversity. Tests whether contextual configs
(Q13) earn their complexity.

**Action-space dimension.** 6–8 weights up to the full vector. Expect BO to
degrade and evolutionary methods to hold up.

**Supply feedback on/off.** Do stationary-world winners survive non-stationarity?

---

## Failure modes to actively hunt

Not "hope these do not happen" — write the checks first.

- **Surrogate exploitation.** Inspect winning configs for signs the optimiser
  found a simulator artifact rather than a real effect. Symptom: excellent in
  sim, mediocre on `TrueWorld`, weight on an implausible dimension.
- **Scale-symmetry waste.** If the ranker's normalisation makes the weight vector
  scale-invariant (Q12), verify optimisers are not burning budget on a direction
  that does nothing.
- **Single-seed overfitting.** Any result that does not survive re-seeding is not
  a result.
- **Trap too easy or too hard.** If every method avoids the bait, or none does,
  the world needs recalibrating before the numbers mean anything.
- **Budget leakage.** Audit that no method is reading `TrueWorld` outside the
  accountant. This is the one bug that would invalidate everything.

---

## Phases

Each phase has an exit criterion. Do not start the next one until it is met.

**P0 — World.** Population, content, response model, ranker, metrics. No
optimisation at all.
*Exit:* exhibit by hand a config that beats the default on the observable proxy
and loses on the true objective. Until the trap demonstrably exists, there is
nothing to benchmark.

**P1 — Harness and baselines.** Budget accounting, protocol runner, B0–B4, U1.
*Exit:* regret curves for all baselines across seeds, with B3 or B4 clearly
beating B1. If the grid baseline wins, the config space is too simple.

**P2 — Surrogate.** `SimWorld`, both fitted and perturbed. Fidelity dial. M1.
*Exit:* the fidelity sweep produces a curve with an identifiable crossover.
An *exploratory* fitted M1 is in `docs/09-phase2.md`. An *exploratory* dialled
sweep (crossover vs B5 at φ≈0.75) is in `docs/12-fidelity.md`. Neither is H3.
Primary comparator for any future confirmatory H3 is B5 (amendment 2026-09-01).

**P3 — Sequential design agent.** M2.
**Deferred.** Exploratory M1 does not beat B5. Do not add an RL outer loop
until confirmatory H1 vs B5 is positive in some fidelity region.
*Exit (if resumed):* M2 measured against every baseline on every metric,
positive or negative. **C4 failed** (`docs/15-c4.md`): per-topic configs do
not buy 20% of headroom, so the contextual extension is not licensed even if
M2 is later resumed for the global case.

**P4 — Writeup.** Figures, reproduction instructions, honest limitations.
*Exit:* someone else can clone the repo and regenerate the headline figure.
Interim writeup: [`13-writeup.md`](13-writeup.md). Headline figure:
`analysis/figures/p2_fidelity.png`.

---

## What a negative result looks like

Worth writing down now, while there is no incentive to be evasive about it.

If M1 and M2 do not beat **B5** at realistic fidelity, the finding is:

> A surrogate fitted from delayed long-horizon labels does not, at achievable
> fidelity, improve on spending the same budget on more of those labels. Here
> is the fidelity threshold that would be required (exploratory bait-blindness:
> the sim must be mostly `V`, φ ≳ 0.75), and here is why a half-right sim is
> discarded by the confirm.

Beating B3 is the uninteresting comparison: B3 cannot see `V`, so shipping
the default already wins. The 2026-09-01 amendment made this the registered
negative-result shape.

That is a genuinely useful result and a more credible piece of work than a
marginal win obtained by tuning our own method against our own benchmark. The
project should be equally happy to publish it. Exploratory evidence already
looks like this paragraph. It is not confirmatory.
