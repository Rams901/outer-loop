# 01 — Objectives

## The problem, precisely stated

A production ranker exposes a policy vector `w` — weights over predicted action
probabilities. Changing `w` changes what the feed optimises for. There is no
retraining involved, so the cost of *trying* a config is near zero and the cost
of *evaluating* one is enormous.

Evaluation is expensive for three separate reasons, and they compound:

**Exposure cost.** Every arm of an experiment consumes real users for real days.
A config that turns out badly harms people while you measure how badly.

**Statistical cost.** The metrics that matter most are the rarest. If a report
occurs three orders of magnitude less often than a like, detecting a meaningful
change in report rate needs orders of magnitude more exposure than detecting the
same relative change in likes. The safety metrics are exactly the ones you can
least afford to measure.

**Latency cost.** The objective is long-horizon retention, but you cannot observe
28-day retention in less than 28 days. So decisions get made on short-horizon
proxies, and the proxy is not the objective. This is the mechanism behind the
walk-back: a config looks good on a two-day proxy and bad once the real signal
arrives.

Given all three, the binding constraint on policy iteration is **user-days of
exposure**, not compute and not ideas. That framing determines everything else
in this project.

## Primary objective

Build a benchmark in which several strategies for searching the config space
compete under an **identical budget of user-days**, against a ground truth we
generated and therefore know the optimum of.

The question the benchmark answers:

> Given a fixed budget of real user exposure, which search strategy finds the
> best config — and how much of that advantage survives when the simulator it
> relies on is wrong?

Note what this is not. It is not "does RL work here." RL is one entrant. Plain
A/B testing over human-proposed values is another, and it is the incumbent we
have to beat honestly.

## Secondary objectives

**Quantify sim-to-real transfer.** Produce a curve: final regret as a function of
simulator fidelity. Somewhere on that curve, simulator-guided search stops
beating direct experimentation and starts losing to it. Locating that crossover
is the most useful thing this project can output, and it is useful whichever
side of it the realistic case falls on.

**Predict walk-backs.** The bidirectional-boost episode is a config that won on
short-horizon metrics and lost on long-horizon ones. Can a surrogate flag that
in advance? A method that finds slightly worse configs but never ships one that
has to be reverted may be the better method.

**Charge for harm done during search.** A strategy that reaches the optimum by
exposing users to terrible configs along the way is not obviously better than a
slower, safer one. Safety violations incurred *during* the search are a reported
metric, not a footnote.

## Non-goals

- **No integration with `x-algorithm`.** Independent repo, independent code. We
  borrow structural facts, not source.
- **No claim to reproduce X's system or numbers.** The world is synthetic and
  says so. Any resemblance to real magnitudes is for realism, not fidelity.
- **Not building a better recommender.** The ranker is fixed and simple. We are
  optimising the *process of tuning it*, which is a different object.
- **Not training a ranking model.** Predicted probabilities are an input we
  simulate, including their errors. Learning them is someone else's problem.
- **No RL-for-its-own-sake.** If Bayesian optimisation wins, the finding is that
  Bayesian optimisation wins.

## Success criteria

Deliberately falsifiable. Each can come out negative and still be a result.

**S1 — The benchmark is well-posed.** Because we generate the ground truth, the
optimal config can be found by brute force offline. Regret is measurable
exactly, not estimated. Without this the whole project is unfalsifiable.

**S2 — The trap is real.** There exists at least one config that scores well on
the observable short-horizon proxy and badly on the hidden long-horizon
objective, and we can demonstrate a naive optimiser walking straight into it. If
we cannot build this, the world is too easy and needs redesigning.

**S3 — Something beats the incumbent.** At least one method beats the
human-style A/B grid at equal user-day budget, by a margin larger than seed
noise. If nothing does, that is a genuine and publishable negative result about
the value of simulation here.

**S4 — The fidelity curve exists.** We can plot regret against simulator
fidelity and identify the crossover point. This is the headline figure.

**S5 — It runs on a laptop.** If a single benchmark run takes a GPU cluster, no
one will reproduce it and it will not function as a benchmark.

## Why this is a plus for the For You series

The series so far has been *reading* a production system. This is the first
piece that *proposes* something, which is a different and higher-value signal:
it moves from "I can navigate a large codebase" to "I can identify the expensive
part and design an experiment against it."

It also follows directly from where post 2 ends. That post closes on the claim
that propose-randomise-observe-correct is a bandit with people inside the policy.
This project is that claim taken seriously enough to test. The honest version of
the follow-up is not "I built a better X algorithm" — it is "I built a small
world where I could measure whether simulation helps, and here is where it stops
helping."

The credibility comes from the ground truth being ours. Every number is
checkable, nothing depends on access to proprietary data, and a negative result
is still a result.
