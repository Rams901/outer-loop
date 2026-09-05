# 02 — Open questions

Decisions to settle before writing code. Marked **[R]** where there is a clear
recommendation, **[FORK]** where the choice genuinely changes the project and
should be made deliberately.

---

## A. Framing

### Q1. Which loop is the RL problem — the ranker, or the config tuner? **[R]**

There are two nested control problems and conflating them is the single most
likely way to waste months.

The *inner* loop is the recommender: given a user and a candidate set, produce a
ranking. The *outer* loop is the tuner: given observed metrics, propose a new
weight vector.

Humans operate the outer loop. That is the thing this project is about.

**Recommendation:** the outer loop. The inner ranker stays fixed, simple, and
deterministic — a faithful re-implementation of weighted scoring with
normalisation. Making the ranker learnable adds a research project underneath a
research project and destroys our ability to attribute results.

### Q2. Is the deliverable a benchmark or a method? **[R]**

**Recommendation:** benchmark first. A method with no benchmark cannot be
evaluated, and the benchmark is the more durable artifact. Methods then get
added as entrants. This also protects against the failure mode where we build a
clever agent and have no way to show it is better than a grid search.

### Q3. Single objective or Pareto front? **[RESOLVED]**

**Decision: one scalar objective — long-horizon retention — subject to hard
safety floors that cannot be traded away.**

Rejected alternatives: a pure weighted scalarisation across retention,
engagement and safety (assumes away the central question by letting safety be
bought with engagement), and a Pareto front (more honest about the tradeoff, but
makes "which method won" unanswerable, which the benchmark needs).

Consequence: safety is a constraint, not a term. A config breaching a floor is
invalid regardless of how good its retention is. This also settles Q14 in favour
of hard constraints.

---

## B. Ground-truth realism

### Q4. What is the minimum complexity that keeps the problem non-trivial?

If every user responds the same way, one global config is trivially optimal and
there is nothing to search. The world needs enough heterogeneity that the
optimum is a genuine compromise. Open: how much is enough, and how do we verify
we have it rather than assuming it?

Proposed check: measure the gap between the best global config and the best
per-segment config. If that gap is near zero, the world is too simple.

### Q5. Must the ground truth contain a Goodhart trap? **[R]**

**Recommendation:** yes, by construction. If maximising the observable proxy also
maximises the true objective, the entire project is vacuous — any optimiser
wins and simulation fidelity does not matter.

The trap needs to be built deliberately: some content must be engaging in the
short term and corrosive in the long term. See `03-synthetic-data.md`.

### Q6. How do we set base rates? **[R]**

`param.rs` contains a comment explaining that report weights are large *because*
reports are rare — roughly a thousand times rarer than likes — so the weight
exists to let a low-probability prediction affect ranking at all. The same
comment explicitly rejects reading weight ratios as count equivalences.

**Recommendation:** make the spread of base rates a first-class design
constraint, spanning at least three orders of magnitude. This is what forces the
statistical cost of evaluation to be realistic, and it is what makes the search
space badly scaled in an interesting way.

### Q7. Do we model supply-side feedback? **[FORK]**

In reality, today's ranking changes what authors post tomorrow. This is the
single biggest reason real recommender simulators fail: they model users
reacting to content but treat the content supply as exogenous.

Including it is more realistic and much harder — the world becomes non-stationary
and the "true optimum" becomes time-dependent, which complicates S1.

Suggested resolution: build the world with feedback **off** by default so the
optimum is well-defined, and add it as an ablation that specifically tests
whether methods that looked good in the stationary world survive. That keeps the
headline result clean while still probing the realistic failure mode.

---

## C. Simulator fidelity

### Q8. How do we represent "the simulator is wrong"?

This is the core mechanic and needs more than one knob, because different kinds
of wrongness have different consequences. Candidates:

- **Functional-form error** — surrogate uses a different response shape
- **Missing features** — surrogate cannot see the attribute that drives the trap
- **Miscalibrated base rates** — especially on rare actions
- **Stale population** — fitted on an older user distribution
- **No feedback loop** — surrogate treats supply as fixed when it is not

Open question: do these collapse onto a single scalar "fidelity" axis for the
headline plot, or do we need a small taxonomy with separate curves? Suspect the
latter is more honest and the former is more communicable.

### Q9. Fit the surrogate from logs, or hand-specify a wrong one? **[R]**

**Recommendation:** both, for different purposes.

Fitting from logged data is the realistic path and exercises the whole pipeline
including off-policy issues. Hand-specifying a perturbed copy of the ground truth
gives a clean, monotone fidelity dial for the headline curve. The two should
agree qualitatively; if they do not, that is itself informative.

### Q10. What is the fidelity metric?

Several plausible definitions and they do not agree:

- Calibration error on held-out impressions
- Rank correlation between simulated and true config values
- Regret transfer — how much worse is the sim's argmax under the true world

Rank correlation over configs is probably closest to what we care about, since
a surrogate can be badly calibrated in absolute terms and still order configs
correctly. Needs deciding before we start reporting numbers.

---

## D. Optimisation

### Q11. How large is the action space? **[R]**

The real vector is roughly two dozen weights. Searching that under a realistic
exposure budget is not a solved problem and is not a good first target.

**Recommendation:** start with 6–8 weights covering the interesting structure —
a couple of common positive actions, one rare strong positive, one rare strong
negative, one continuous signal, one contextual boost. Scale up as an ablation.

### Q12. Should we quotient out the scale symmetry? **[R]**

Because scores are normalised before comparison, multiplying the entire weight
vector by a positive constant should not change the ranking. If that symmetry is
present, a naive optimiser wastes a dimension of its budget exploring a direction
that does nothing.

**Recommendation:** verify the symmetry holds in our ranker, then search on the
unit sphere or in a normalised log-space. Worth stating explicitly in the
writeup — it is a concrete, non-obvious consequence of reading the scoring code.

### Q13. One global config, or contextual per segment? **[RESOLVED]**

**Decision: global config first. Contextual is a later extension, not part of
the headline benchmark.**

The cost of this decision is real and should be stated plainly: with a single
global config, the problem is low-dimensional black-box optimisation, and
Bayesian optimisation may simply win. If it does, that is the honest finding for
this phase, and it is the finding that makes the contextual extension worth
doing rather than assumed.

Consequence for sequencing: the contextual extension is where RL has its
clearest structural advantage, so it moves from "maybe" to "the planned follow-up
if phase 1 shows the global case is saturated by BO."

### Q14. Are guardrails constraints or penalties? **[RESOLVED]**

**Decision: hard constraints**, following from Q3. A config breaching a safety
floor is invalid regardless of its retention, and the budget spent discovering
that is not refunded.

---

## E. Protocol

### Q15. What exactly is one unit of budget?

"User-days of exposure" needs a precise definition covering: users in holdout vs
treatment, the ramp period, and whether observing a delayed metric costs
anything after exposure ends. Ambiguity here makes cross-method comparison
meaningless.

### Q16. How many seeds constitute a result?

Two sources of randomness: the instantiation of the ground truth, and the
stochastic rollouts within it. Both need replication. Suspect ≥20 ground-truth
seeds, but this should be set by a pilot variance estimate rather than picked.

### Q17. Do we pre-register? **[R]**

**Recommendation:** yes. We are building the benchmark *and* competing in it,
which is a conflict of interest. Write down the primary metric, the number of
seeds, and the success thresholds before running anything, and keep the file in
git history so the timestamps are visible.

---

## Status

**Resolved:** Q3, Q13, Q14. Q15 is pinned *for the phase-1 pilot only*
(24k user-days, 800×2-day looks, treatment charged, oracle free) in
`docs/08-phase1.md` — not a stage-2 lock.

**Still open and deliberately deferred:** Q7 (supply feedback — planned as an
ablation, off by default). Q4, Q6, Q8, Q9, Q10, Q15 and Q16 are implementation
questions whose answers depend on the phase 0 pilot; they are resolved in
`PREREGISTRATION.md` stage 2 rather than by argument.
