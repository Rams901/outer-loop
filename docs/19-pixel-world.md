# 19 — Pixel world (pygame)

A 21-day **renderer** of a real `TrueWorld` rollout. Not a second simulator.
Not a city the ranker walks through.

```bash
pip install pygame
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.pixel_world
# optional: --seed 0 --watch 96
```

Tab switches **default** vs **engagement trap** (paired rollouts, same seed).
Click a cell = one user. ← → = day. Space = play through day 21.

## Why not the city-grid suggestion

Strategy A (wandering 8-bit citizens, content balloons, same-colour users
clumping into a filter bubble) is a good *game*, and a **lie about this
ranker**.

- Users do not move on a map. Ranking is `score = w · p̂` over a candidate
  set. Geography is not in the model.
- There is **no follow graph**. “Followers vs non-followers” is mapped to
  **in-interest vs OON**: relevance ≥ vs < the catalogue median. That is the
  honest analogue.
- Filter bubbles here are **feed composition** (bait share, topic mix), not
  avatars pooling in a plaza. Same-topic users sit together in the grid only
  as a **legend** (stratified sample), so you can see whether a whole colour
  goes gray (churn) or grows a rose pip (bait-heavy feed).

docs/07 already warned: mixing timescales and drowning in 4,000 sprites
confuses people. We watch **~96 users × 21 days**, plus a microscope on one
feed.

## What you are looking at

| Signal | Where | Meaning |
|---|---|---|
| Cell colour | topic | Primary interest (4 topics) |
| Brightness | satisfaction | Hidden state → p(return tomorrow) |
| Gray cell | churned that day | Did not open the app |
| Rose pip | bait share ≥ 0.4 | Trap content in that user’s top-12 |
| Feed tiles | rose / green / slate | bait / in-interest / OON |
| like / dwell / hide | on the tile row | Sampled actions that day |
| Cohort return / bait / engage / sat | header | Population series (C1 chart, live) |
| V and proxy | header | Full-horizon retention vs 2-day likes+dwells |

Compare default vs engagement on the same users: engagement should light more
rose pips and, over days, more gray cells — that is the walk-back, visible.

**LinkedIn / screen-record:** click one user, Space through 21 days, Tab to
the trap, play again. Paste-ready caption: [`20-linkedin-post.md`](20-linkedin-post.md).

## Feasibility

pygame is enough: no physics, no pathfinding, one blit loop. The cost is the
two numpy rollouts (seconds), not the viewer. Scaling to 4,000 on-screen
sprites is possible and still the wrong story.
