# 07 — Tracing and visualization

The rollout today returns one `RolloutResult`. Everything interesting happens
inside the day loop and is thrown away. If we want animations that stay true as
the project grows, **the simulation must emit a trace**. Renderers consume the
trace. They do not re-implement the math.

This document locks the causal chain (what to show), the trace schema (how to
codify it), and the rendering options (how to draw it).

---

## The chain, named

There are two timescales. Mixing them in one animation is why these explainers
usually confuse people. Split them.

### A. Init (once per world seed)

Happens in `TrueWorld.__init__`, before any ranking.

| Name | What it is | Where |
|---|---|---|
| Population | users with interests, propensities, sensitivity | `_make_population` |
| Catalogue | posts with quality, bait, toxicity, topics, cold-start | `_make_catalogue` |
| True logits | `base + propensity + rel + quality + bait + tox` | `_precompute_true_logits` |
| Predicted probs | `sigmoid(true_logit + post-level noise)` | `_pred_probs` |
| Satisfaction | hidden scalar, starts at 0.58 | `rollout` |

The ranker never sees bait, quality, or satisfaction. It only sees predicted
probabilities. That sentence is the whole trap, and the animation has to make it
visible.

### B. Inner loop (one user, one day) — this is “score shifted”

This is the thing to animate in detail. One named user, a handful of candidates,
two weight vectors side by side.

```
candidates sampled
        │
        ▼
predicted p̂  (what the ranker sees)
        │
        ▼
score = w · p̂     ← the only place weights act
        │
        ▼
top-k feed, slots 0..11
        │
        ▼
true p × position decay  (what actually happens)
        │
        ▼
actions sampled
        │
        ▼
Δsatisfaction from quality − bait  (hidden)
        │
        ▼
p(return tomorrow) = sigmoid(6 × (s − 0.48))
```

Definitions used on this path:

- **Candidate set** — 48 posts drawn that day. Not the whole catalogue.
- **Score** — `w · p̂`. Not a learned model. Changing `w` changes ranking without
  retraining anything.
- **Feed** — top 12 by score. Slot 0 is most visible (`position_decay ** slot`).
- **Proxy** — likes + dwells per impression on days 0–1. Observable fast.
- **V** — mean daily return on days 1–20. Hidden from the ranker; the benchmark
  score.

### C. Outer loop (not built yet)

Phase 1+: propose a config, spend user-days, observe a noisy proxy, maybe see V
later, commit. Same trace schema, new event types (`experiment_started`,
`budget_debited`, `commit`).

---

## Codify this: a trace, not a video

Do not drive an animation off the 4,000-user numpy arrays. Pick **one canonical
scenario** (one user, one day, ~8 candidates, both configs) and record every
intermediate as JSON. The 4,000-user run stays the source of C1 numbers; the
scenario is a microscope on the same equations.

Properties that matter:

1. **Determined by seed.** `world_seed`, `user_id`, `day`, `rollout_seed` fully
   specify it. Re-running the tracer regenerates the file. Checked-in traces are
   fixtures, not art.
2. **Versioned schema.** `rl_opt.trace.v1`. Additive fields as phases land;
   never silently rename.
3. **Two consumers, one log.** A Remotion video and a matplotlib figure read the
   same file. If they disagree, the tracer is wrong, not the renderer.
4. **Cheap in the hot path.** Full rollouts stay aggregate-only. Tracing is an
   explicit `trace_scenario(...)` call, not a flag that logs 4,000 users.

### Schema sketch (`rl_opt.trace.v1`)

```json
{
  "schema": "rl_opt.trace.v1",
  "world_seed": 0,
  "rollout_seed": 7,
  "user_id": 17,
  "day": 0,
  "user": {
    "interests": [0.6, 0.2, 0.1, 0.1],
    "sensitivity": 1.1,
    "satisfaction_before": 0.58
  },
  "configs": {
    "default":     { "weights": [0.5, 5.0, 0.05, 2.0, -43.2, -234.0] },
    "engagement":  { "weights": [2.2, 3.5, 1.4, 1.5, -32.0, -90.0] }
  },
  "candidates": [
    {
      "post_id": 41,
      "quality": 0.81,
      "bait": 0.08,
      "toxicity": 0.04,
      "cold_start": false,
      "relevance": 0.72,
      "true_p":  { "like": 0.14, "reply": 0.04, "dwell": 0.22, "share": 0.02, "hide": 0.002, "report": 0.00005 },
      "pred_p":  { "like": 0.16, "reply": 0.03, "dwell": 0.19, "share": 0.02, "hide": 0.003, "report": 0.00008 },
      "score_terms": {
        "default":    { "like": 0.08, "reply": 0.15, "dwell": 0.01, "share": 0.04, "hide": -0.13, "report": -0.019, "total": 0.13 },
        "engagement": { "like": 0.35, "reply": 0.10, "dwell": 0.27, "share": 0.03, "hide": -0.10, "report": -0.005, "total": 0.65 }
      },
      "rank": { "default": 2, "engagement": 7 },
      "shown": { "default": true, "engagement": false }
    }
  ],
  "feed": {
    "default":     { "slots": [41, 8, 12], "mean_bait": 0.05, "delta_s": 0.012 },
    "engagement":  { "slots": [77, 41, 3], "mean_bait": 0.41, "delta_s": -0.018 }
  },
  "satisfaction_after": { "default": 0.592, "engagement": 0.562 }
}
```

`score_terms` is the piece worth animating: each action’s `w_i * p̂_i`, then the
sum, then the sort. That is “how the score shifted.”

A second, smaller artifact is a **horizon strip**: daily `{proxy, V, bait_share,
mean_s}` for both configs over 21 days, taken from the full rollout. That is
the C1 chart. It does not need per-candidate detail.

---

## Rendering options

All of these assume the trace exists. None of them should compute scores
themselves.

| Option | Good for | Cost | Stays honest as the project grows? |
|---|---|---|---|
| **1. Trace JSON + pytest snapshots** | Catching silent formula changes | Low | Yes — this is the foundation either way |
| **2. Matplotlib / Plotly figures** | Phase reports, C1 chart, paper | Low | Yes. `analysis/*.png` regenerated from the trace |
| **3. Remotion (existing `xalgo-video`)** | LinkedIn, the “one user, two feeds” story | Medium | Yes if it *reads the JSON*. No if scenes are hard-coded numbers |
| **4. Streamlit / small HTML viewer** | Your own debugging: pick user, pick day, step through | Medium | Best for *doing* the research, weak as a published artifact |
| **5. Manim** | Math-textbook motion | High | Off-label. We already rejected it for diagrams |
| **6. Animate all 4,000 users** | Nothing | High | Lies by drowning. Do not do this |

Option 3 is the same craft as the orq.ai reference: staged reveals of a
structure that actually happened. Option 2 is what phase 1–4 reports need.
Option 4 is what you will want the week the trap looks “too easy” again.

### What not to do

- Hard-code the +20% / −0.23 numbers into a video. Recalibration will desync it.
- Log every impression in the 4,000 × 21 loop. That is a dataset, not a story.
- Start in After Effects. The numbers will move; the composition will not.

---

## Recommended approach

Three layers, in this order:

**L1 — Tracer in `rl_opt`.** `trace_scenario(world, user_id, day, configs) -> Trace`.
Fixture checked in at `analysis/traces/c1_user.json`. A test asserts the schema
and that `score_terms.total == sum(terms)`.

**L2 — Two figures from the same trace.** (a) candidate score breakdown, default
vs engagement, stacked by action. (b) 21-day horizon: proxy vs V vs bait share.
These belong in `docs/06` and later phase reports. No new video tooling.

**L3 — Remotion reads L1.** One composition: init → this user → terms light up →
two feeds reorder → satisfaction ticks → C1 numbers at the end, sourced from the
trace. Palette stays the xalgo-video system so the series matches.

Do L1+L2 before L3. A video of a story we cannot regenerate is worse than a
static figure we can.

## Built in this pass

```bash
python -m rl_opt.export
```

writes:

| File | Role |
|---|---|
| `analysis/traces/c1_user.json` | one user, 48 candidates, both configs, `score_terms` |
| `analysis/traces/c1_horizon.json` | 21-day series for the full population |
| `analysis/traces/viewer.html` | stepper: click a candidate, see `w · p̂` on both sides |
| `analysis/figures/c1_score_terms.png` | disputed posts, two totals |
| `analysis/figures/c1_horizon.png` | return vs bait share vs like+dwell |

Open the HTML as a file. JSON is inlined (no server). This is the debugging
surface until a Streamlit app is worth it.

Canonical user for world seed 0 is **2469**. On that user's day-0 candidate
set, default feed mean-bait is 0.00 and Δs is +0.011; engagement feed mean-bait
is 0.83 and Δs is −0.023. That is more extreme than the *population* C1 mix
(~33% bait) on purpose: the picker maximises rank disagreement so the
microscope is readable. C1 numbers stay the population ones.

## Streamlit later

Do not add the dependency until phase 1 has a budget to inspect. The app should
be a thin shell over APIs that already exist:

```python
world = TrueWorld.generate(params, seed)
pick = pick_canonical_user(world)          # or a user_id slider
trace = trace_scenario(world, pick.user_id, pick.candidate_ids)
result = world.rollout(weights, seed=rollout_seed)
```

Widgets: world seed, user id, day, config A vs B. Output: the same two figures
plus the candidate table. If Streamlit is missing, `viewer.html` is the
stand-in. No second scoring implementation inside the app.

As phases land, the same schema grows:

| Phase | New events |
|---|---|
| 0 (now) | init, candidate scores, feed, Δs, C1 aggregates |
| 1 | budget debit, experiment receipt, delayed V |
| 2 | SimWorld prediction vs TrueWorld, fidelity |
| 3 | agent propose / commit |

---

## Canonical scenario (when we build this)

Not “user 0.” Pick the user in seed-0 whose day-0 candidate set contains at
least one high-bait and one high-quality post, where default and engagement
**disagree on rank order**. That disagreement is the picture. If no such user
exists, the trap is not happening at the ranking step and the world needs
another look — which is itself a reason to have the tracer.
