# 18 — Business value, LinkedIn draft, new product vs X

Not a confirmatory report. Numbers below are from this synthetic benchmark
(`docs/17-h1.md`, `12`, `16`, `15`). They do **not** transfer as For You
weights. The transferable object is a **decision protocol**: what to spend
user-days on, and when a simulator is lying.

---

## LinkedIn post draft

X’s For You ranker does not pick tweets by training a new model every time the
product changes. It multiplies predicted probabilities by a weight vector —
`0.5` for a like, `5` for a reply, `−234` for a report. Changing what the feed
optimises for is changing a number. The open-source trail shows how those
numbers actually move: a mutual-follow reply boost was tried at 5, 10, 15 and
20, rolled out at 20, then walked back to 15.

That loop is a search procedure with people inside it. I built a small
synthetic world where the inner ranker is frozen (`score = w · p̂`) and the
outer loop — the config tuner — is the experiment. Question: can offline
simulation reduce live A/B tests for those weights, and can we tell when the
sim stops being trustworthy?

What we tried, and what it proved:

**1. Search the two-day proxy** (grid, random, GP, ES).  
Every method beat the default on short-horizon engagement and **lost on
21-day retention**. Walk-back rate 5/5. The trap was built in: bait is
genuinely likeable and quietly corrosive. Conclusion: the bidirectional-boost
failure mode is not a one-off. If you maximise the metric you can see
tomorrow, you will ship a reversal.

**2. Spend the same budget on delayed retention labels** (B5).  
Eleven full-horizon measurements, ship the best *measured* config. About
**+0.10** retention vs doing nothing. Conclusion: the expensive metric *is*
the product. Waiting beats clever search on the wrong number.

**3. Fit a simulator on those labels and extrapolate** (M1 vs B5).  
Confirmatory, N=50, locked seeds, one allowed retune frozen in advance.
Wilcoxon p=0.013, median regret edge −0.025, bootstrap CI **touches 0**.
Mean retention **0.562 vs 0.560**. Conclusion: a model of the labels does
not replace buying more of them. Powered for a 0.10 gap in regret; we saw a
quarter of that. Practically a tie.

**4. Dial how wrong the sim is** (fidelity / Q8).  
A sim that cannot see that bait hurts retention is discarded by a live
confirm — you ship default. Crossover vs honest labels sits near “mostly the
true objective” (exploratory bait-blindness φ ≈ 0.75). Stale user tastes and
miscalibrated rare-event predictions barely matter. Conclusion: wrongness is
not one number. Panic if the sim is **blind to emptiness**. Don’t panic-rebuild
it for every report-calibration bug.

**5. Per-segment weights** (C4).  
Best per-topic config vs best global: ~0 of headroom. Conclusion: don’t staff
a contextual tuner until you can show the global config is a real compromise.

Business value, one line: **fewer walk-backs, experiments spent on delayed
retention, a rule for when the sim is not allowed to ship.** Not a new ranking
model. Not “replace A/B testing.”

A simulator cannot replace Home Mixer experiments. It can stop you running
the ones that only win on two-day engagement. The ones you *do* run should
buy delayed retention — not a surrogate of those same labels. You know the
sim is lying when it cannot see the thing that makes people stop coming back.

(Standalone synthetic benchmark, pre-registered, not X production data.)

---

## Theories we ran → analysis they licensed

| Theory | Entrant / dial | What we measured | Conclusion that survives |
|---|---|---|---|
| Proxy search Goodharts | B1–B4 | Proxy up, `V` down, bait up | Do not maximise 2-day likes/dwells. Walk-backs are the product cost. |
| Delayed `V` is the channel | B5 vs B0 | ~+0.10 retention, 0 walk-backs | Pay to wait. Same budget, honest metric. |
| Fitted sim reduces tests | M1 vs B5 (H1, N=50) | +0.002 V, CI includes 0 | Does **not**. Surrogate ≈ extra interpolation on the same 11 labels. |
| Sim helps iff mostly `V` | Fidelity φ, Q8 axes | Half-bait = default; φ=1 tiny edge over B5 | Confirm-on-live is the control. Missing-feature error is the lethal one. |
| Heterogeneity needs segments | C4 | Gap / headroom ≈ 0.004 vs 0.20 | Global `w` first. Contextual RL not licensed here. |

Needle that moves: **measure delayed retention.**  
Needle that does not: **simulate instead of measuring.**

---

## New product / new website: the plan

A greenfield recommender has **no default that has been A/B’d** and **no
delayed `V` yet**. That is the worst time to “start with a sim.” Our M1 only
looked like a method because it was already fed charged full-horizon labels —
that spend *is* A/B.

**Day 0.** Ship a conservative default (do not overweight cheap engagement).
You will only have short-horizon proxies. Treat them as **unsafe to
maximise.** B1–B4 are the warning: the first optimiser pointed at likes will
look like a win and rot return.

**First live tests.** Not a 50-arm search. A **small delayed-V screen** (B5):
default + a few principled alternatives (quality tilt vs engagement tilt).
Hold for the real horizon. Ship the best **measured** `V`. That is a good
start. It is A/B, honest about latency.

**As more tests accumulate.** Logs under one policy still do not identify
`V(w')` for other weights. More A/B does **not** automatically unlock a
trustworthy sim. It unlocks **more rows in B5’s table.** A surrogate starts
to pay only when it is mostly the same objective *and* you still confirm the
winner live. Until then, extra tests should buy **labels**, not extrapolations.

**Mature.** Use the sim to **kill** candidates (don’t waste traffic on
bait-shaped weights), not to **crown** them. Confirm stays 1–2 live labels.
If sim and confirm disagree, ship default or the measured winner — never the
sim’s favorite. That is M1 at low fidelity: the confirm is the product.

Stack, in order: **default → delayed-V grid → optional sim as a filter.**  
Never: **sim first, tests later.**

---

## X For You: what simulation can do, and when we know it’s wrong

You cannot paste synthetic `DEFAULT_WEIGHTS` or M1’s 0.562 into `param.rs`.
August Home Mixer is already closer to a quality-shaped vector than to our
beatable B0. We *moved* B0 off production on purpose so the benchmark had
headroom. Claiming For You is “under-tuned by 0.10” from this world would be
a category error.

**What simulation can do for For You**

1. **Stop walk-backs before ramp.** Exhibit already in-repo: boost 20 → 15.
   Offline, ask: does this config win the 2-day proxy and lose delayed
   retention? If yes, do not ramp. That is C1 as a ritual, not a plot.
2. **Change what traffic is for.** Retuning Favorite / Dwell / bidirectional
   reply: spend user-days on delayed-retention labels of a short list, not on
   a fitted ranker of configs. The 0.10 vs 0.002 split is the budget story.
3. **Filter, don’t crown.** A Phoenix replay / value model / offline score
   can drop configs that look like the trap. Only a live delayed-`V` confirm
   may ship.
4. **Don’t open a second control loop.** C4: per-segment weights didn’t pay.
   An RL agent that picks the next experiment is not justified by these
   numbers.

**When we know the sim is wrong**

- **Lethal:** it cannot see what makes people stop coming back (bait,
  empty dwell, “engagement” that does not retain). φ ≈ 0 on that axis: it
  recommends the trap; **only confirm ships.** If confirm rejects, you
  learned the sim is blind — that is the product signal.
- **Mild (here):** stale population, hide/report miscalibration. Config
  ranking can still be OK. Don’t rebuild the world model for every rare-event
  calibration ticket.
- **Operational test:** sim-best vs default on live delayed `V`. If the
  sim-best loses, you ship default and you have a fidelity failure, not a
  tuning failure.

**Pitch for an internal For You note**

> Offline simulation cannot replace Home Mixer experiments. It can stop us
> from running the ones that only win on two-day engagement. The experiments
> we do run should buy delayed retention, not a surrogate of those same
> labels. We know the sim is lying when it cannot see the thing that makes
> people stop coming back.

---

## What this is not

- Not a recommendation to change production FavoriteWeight / DwellWeight.
- Not “skip A/B on a new site.”
- Not a license for M2 / PPO / per-user configs from these results.
- Not mixing pre-recalibration tables (`docs/06`–`08`) with `09`/`12`/`17`.
