# 04 — What defines the RL environment

## Two nested loops, and only one of them is ours

The most common way to get this project wrong is to build the wrong MDP. There
are two, they are easy to confuse, and they have almost nothing in common.

### Inner loop — the ranker (fixed, not learned)

Given a user and a candidate set, score every candidate as a weighted
combination of predicted action probabilities, normalise, sort, return a feed.

This is a deterministic function of the weight vector. It is a faithful but
small re-implementation of `compute_weighted_score`, including the normalisation
by positive, negative and total weight sums. **We do not learn it.** It is the
thing being configured, not the thing being optimised.

### Outer loop — the tuner (ours)

Given the history of configs tried and metrics observed, propose the next config
to test, and eventually commit to one.

Humans run this loop today. It is the subject of the project.

Keeping these separate is what makes results attributable. If the ranker were
also learning, no observed improvement could be assigned to either loop.

---

## The outer problem, stated

- **State** — configs tried so far, metrics observed (some still pending due to
  reporting delay), budget remaining, and optionally segment context
- **Action** — propose a config to evaluate, and choose how much exposure to
  give it
- **Transition** — the world runs that config on that slice for that duration
- **Observation** — short-horizon proxies immediately; long-horizon truth after
  a delay; both noisy
- **Cost** — user-days consumed, debited from a fixed budget
- **Termination** — budget exhausted, at which point the agent must **commit** to
  a single config
- **Score** — the true long-horizon value of the committed config, plus safety
  violations accumulated during the search

Committing at the end matters. Scoring the best config ever *observed* rewards
lucky noise; the agent has to decide, under uncertainty, which one to ship.

## Call it what it is

This is **budgeted black-box optimisation with expensive, noisy, delayed and
risky evaluations**. It is much closer to bandits and Bayesian optimisation than
to deep RL over user trajectories.

Saying so is a feature, not a concession. It means the baselines are strong, and
if an RL method wins it wins for an identifiable reason rather than because the
comparison was rigged.

### Where RL genuinely earns its place

Three framings, in increasing order of interest:

**1. Contextual configs.** Choose different weights per user segment rather than
one global vector. The action space becomes large and structured, and
context-dependence is exactly what contextual bandits are for. This is also
closer to how feature switches actually bucket users. Gated on Q13.

**2. Sequential experiment design.** The agent's action is *which experiment to
run next* — which config, on how many users, for how long. That is a genuine
sequential decision problem over the experimentation process itself, with an
explore/exploit structure that a one-shot optimiser does not have. This is the
most novel framing and the proposed headline.

**3. Model-based planning.** Learn the world model from logs, plan against it,
deploy, observe, correct the model. This is Dyna-style model-based RL and it is
the direct formalisation of "simulate instead of A/B test." It is best treated as
the *mechanism inside* framing 2 rather than a separate entrant.

Proposal: build framing 2 as the headline with 3 as its engine, and hold 1 as an
extension once the basic benchmark is sound.

---

## The two-world design

This is the central architectural decision, and it is what makes sim-to-real gap
measurable rather than hand-waved.

**`TrueWorld`** — the hidden ground truth from `03-synthetic-data.md`. Only the
benchmark harness may call it. **Every call debits the budget.** The agent never
touches it directly.

**`SimWorld`** — the surrogate. The agent may query it as much as it likes, for
free. It is either fitted from logs the agent has already paid for, or a
deliberately perturbed copy of `TrueWorld` with a fidelity parameter.

The gap between them is the object of study.

Enforcing the budget at the API boundary is the key trick: the agent *cannot*
cheat by peeking at the ground truth, because the only path to it goes through
an accountant. Any method that wants more true samples must give up something
else. This turns "be honest about your sample budget" from a discipline into a
structural guarantee.

---

## Interface sketch

Not final, but the shape it needs:

- `TrueWorld.rollout(config, n_users, n_days, seed) -> Receipt` — debits budget,
  returns short-horizon metrics immediately and a handle for delayed metrics
- `Receipt.poll()` — long-horizon metrics, available only once enough simulated
  time has passed
- `SimWorld.rollout(...)` — same signature, free, no delay
- `Budget` — user-days remaining; raises rather than silently overspending
- `Agent.propose(history) -> Experiment` and `Agent.commit() -> Config`

The symmetry between `TrueWorld` and `SimWorld` signatures is intentional: an
agent should be able to run against either without knowing which, so
"how well does this agent do with a perfect simulator" is a free ablation and
an upper bound.

---

## Guardrails

Safety floors — report rate, block rate, hide rate — are **hard constraints**,
subject to Q14. A config breaching one during evaluation is force-reverted, and
the budget spent getting there is not refunded.

This is modelling something real: a bad rollout costs you the exposure *and* the
harm *and* the time, and you do not get a do-over. Methods that blunder into
unsafe regions should be visibly penalised for it, which requires that the
penalty be structural rather than a term in a reward.

---

## The reward specification problem

The hardest and most easily fudged part.

If the agent optimises the true long-horizon objective directly, we have assumed
away the central difficulty of the real problem — nobody knows the true
objective, which is exactly why proxies get used and why walk-backs happen.

So the setup must be:

- the agent **optimises** a proxy it can observe (short-horizon engagement,
  possibly with delayed retention arriving late and partially)
- the benchmark **scores** it on the hidden true objective

The agent never sees its own score until the run ends. This is uncomfortable and
correct: it reproduces the actual epistemic situation of a ranking team, and it
is the only way the Goodhart trap can function as a test rather than a
decoration.

---

## Non-stationarity

Default: stationary world, so the optimum is well-defined and regret is exact
(S1). Supply-side feedback and population drift enter as **ablations**, where the
question becomes narrower and more interesting: do the methods that won in the
stationary world survive when the target moves? Suspicion is that model-based
approaches degrade fastest, since their surrogate is fitted to a world that no
longer exists.

---

## Open items

- Whether the agent chooses exposure size and duration, or only the config. The
  former is richer and closer to real experiment design; it also enlarges the
  action space considerably.
- How to represent pending-but-unobserved experiments in the agent's state.
- Whether to allow early stopping of a running experiment, which is what real
  teams do and which changes the problem meaningfully.
- Whether `SimWorld` refits after each true observation (online) or once per
  phase (batch). Online is more realistic and much slower.
