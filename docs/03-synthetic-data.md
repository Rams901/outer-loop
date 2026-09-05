# 03 — What defines the synthetic data

## The framing that matters most

We are not generating a dataset. We are generating a **world**, and the dataset
is whatever that world emits when a policy interacts with it.

This distinction is the whole project. A fixed dataset can only tell you about
the policy that produced it. The question here — "what happens if we change the
weights?" — is counterfactual, so the data has to be *policy-dependent by
construction*. Change `w`, get different impressions, different engagement,
different retention, different logs.

It also means "the synthetic data" is really four things:

1. a **population** of users
2. a **catalogue** of content and authors
3. a **response model** mapping (user state, post, position) to behaviour
4. the **logs** that fall out when a policy runs against 1–3

Only the fourth looks like a dataset. The first three are the specification.

---

## 1. Population

Each user carries:

**Latent interests** — a vector in a modest number of topic dimensions.
Determines relevance against post topics.

**Action propensities** — per-user offsets on each action's base rate. Some
people reply constantly and never like; some report readily; most do neither.
This heterogeneity is what makes a single global config a compromise rather than
an obvious answer, which is the point of Q4.

**Sensitivity** — how strongly content quality moves their satisfaction, and how
much low-quality content they tolerate before disengaging. Varies across the
population, so the same config helps some users and harms others.

**Activity** — sessions per day, posts consumed per session.

**Hidden satisfaction state** — a scalar that evolves with what they are shown
and drives their probability of returning tomorrow. This is the only state with
memory, and it is where the long-horizon objective lives.

The satisfaction state is the load-bearing component. Without persistent state
there is no distinction between short-horizon and long-horizon metrics, no
walk-back phenomenon, and no reason simulation would be hard.

## 2. Content and authors

**Authors** carry a quality parameter, a topic, an audience size drawn from a
heavy-tailed distribution, and a cold-start flag (few impressions, uncertain
quality — the population that Thompson sampling exists to handle in the real
repo).

**Posts** carry a topic vector, a quality score, a **bait score**, and a
toxicity score. Posts arrive over time.

The bait score is the trap. See below.

## 3. Response model — the ground truth

Given a user in their current state, a post, and the slot it occupies, the world
produces:

- a probability for each modelled action (like, reply, repost, share, profile
  click, dwell, hide/not-interested, report)
- a realised sample from those probabilities
- an update to the user's satisfaction state
- eventually, a return/no-return decision for the next day

Three properties this model must have:

### Base rates spanning orders of magnitude

Likes are common, replies less so, reports vanishingly rare. The spread should
cover at least three orders of magnitude.

This is not decoration. It is the reason the real weights look the way they do —
`param.rs` explains that report weights are large *because* reports are rare, so
that a low-probability prediction can influence ranking at all. Reproducing the
spread reproduces two consequences we need:

- the search space is badly scaled, so naive optimisation in raw weight space
  wastes budget
- rare-event metrics need far more exposure to measure, which makes the budget
  constraint bite where it actually bites in practice

### Position bias

Probability of engagement decays with slot. Without this, ranking quality barely
matters and the config space is flat.

### Prediction is not truth

The ranker does not see true probabilities. It sees *predictions* with error —
biased, heteroskedastic, worse on rare actions and cold-start authors.

This matters more than it looks. The real weights multiply predicted
probabilities, so the interaction between weight magnitude and prediction error
is a real effect: a large weight on a badly-predicted rare action amplifies
noise into the ranking. A world that hands the ranker true probabilities skips
the most interesting source of difficulty.

---

## The Goodhart trap, built deliberately

The world must contain content that is **rewarding in the short term and
corrosive in the long term**. Concretely, high-bait posts should have:

- elevated probability of like and dwell
- little or no elevation in report or hide — it is not rule-breaking, just empty
- a *negative* contribution to satisfaction

The consequence: any config that over-weights likes and dwell will rank bait
highly, look excellent on two-day engagement, and quietly erode retention. That
is the mechanism behind a real walk-back, and it is the thing a surrogate has to
be good enough to catch.

Two design obligations follow.

**It must be findable but not obvious.** If bait is trivially separable, a
surrogate detects it immediately and simulation always wins. If it is invisible,
nothing works and the benchmark is degenerate. Calibrating this difficulty is a
real task, probably requiring a tunable parameter and a pilot study.

**It must be verifiable before any optimisation runs.** Phase 0 exit criterion:
exhibit a specific config that beats the default on the observable proxy and
loses to it on the true objective. If we cannot produce that by hand, the world
needs redesigning.

A useful second trap, closer to the actual bidirectional-boost story: a boost
that helps at moderate strength and hurts past a threshold, because it starts
crowding out content the user would have preferred. That gives a non-monotone
response along one axis — exactly the shape that produced 20 being shipped and
then reverted to 15.

---

## What the world emits

Running a policy against the world produces:

- **Impression logs** — user, post, slot, predicted probabilities, realised
  actions. The off-policy learning substrate.
- **Session logs** — length, depth, when the user left.
- **Daily retention** — returned or not.
- **Aggregate metrics per config** — split into short-horizon proxies available
  quickly, and long-horizon truth available only after a delay.

The delay is deliberate and important. Long-horizon metrics arriving late is
what forces decisions onto proxies, which is what makes the whole problem hard.
A world where truth is instantly observable has no need for simulation.

---

## What generating it ourselves buys

**A known optimum.** We can brute-force the best config offline with unlimited
simulated compute, because it is our world. That gives exact regret rather than
an estimate, which is the foundation of S1 and something no real-data study can
have.

**A controllable fidelity gap.** We can perturb a copy of the ground truth in
specified ways and measure what each kind of wrongness costs.

**Verifiable claims.** Every number in the eventual writeup is reproducible from
a seed by anyone who clones the repo.

---

## Parameters that must be sweepable

Population size · horizon in days · interest heterogeneity · propensity spread ·
bait prevalence · bait detectability · prediction error magnitude · position
bias strength · metric reporting delay · supply feedback on/off

## Sizing

Start small enough to iterate in seconds, not hours — on the order of ten
thousand users over a month of simulated time, with everything scaling from a
single knob. If a benchmark run does not fit on a laptop it will not be
reproduced, and an unreproduced benchmark is not a benchmark.

---

## Open items carried into implementation

- How to calibrate bait detectability so the benchmark is neither trivial nor
  degenerate (needs a pilot).
- Whether satisfaction should be scalar or multi-dimensional — scalar is
  simpler, but a single number may make the trap too easy to infer.
- Whether the true optimum stays well-defined once supply feedback is enabled.
  It probably does not, which is the argument for keeping feedback as an
  ablation rather than a default (see Q7).
