# 20 — LinkedIn post (pygame video)

Paste-ready. Media: screen recording of `python -m rl_opt.pixel_world`,
**one user**, days 0→20, **Tab** to switch default vs engagement trap.

Do **not** claim you ran RL on live For You or shipped `param.rs`. This is a
synthetic benchmark inspired by Home Mixer weights. Confirmatory “RL-style”
search (fitted sim vs honest labels) is in `docs/17-h1.md`.

**Post this:** the **shorter hook** below (with the video). The long “Post
(copy)” is a fallback if you want more detail on LinkedIn.

DEV.to (equations + figures): [`21-devto.md`](21-devto.md). Notes:
[`18-business-value.md`](18-business-value.md). Viewer:
[`19-pixel-world.md`](19-pixel-world.md).

---

## How to record the video (30–45s)

```bash
pip install pygame
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.pixel_world
```

1. Click **one cell** (one user) so the right-hand feed is the story.
2. Stay on **default**. Space = play. Watch tiles, sat bar, gray vs colour.
3. **R** back to day 0. **Tab** = engagement trap. Play again.
4. Optional: pause mid-horizon and point at rose pips (bait) vs green
   (in-interest) vs slate (OON).

Caption on-screen is already the legend. Crop to the inspector panel if the
grid is noisy.

---

## Post (copy)

I simulated how ranking **weights** — the kind X For You keeps in
`param.rs` — change what one person actually sees, day after day, for 21 days.

Not a new neural ranker. The inner model is frozen: `score = w · p̂`. Change a
number (like, reply, dwell, report) and the feed reorders. That is how Home
Mixer already works. The open-source trail even shows the human loop: a
reply boost tried at 5 / 10 / 15 / 20, shipped at 20, walked back to 15.

So I built a small world where that loop is measurable, and a pixel viewer
so you can watch it.

**The video:** one user. Left: a cohort (colour = interest, gray = didn’t
come back, rose pip = bait-heavy feed). Right: that day’s top-12.
Rose = trap content, green = in-interest, slate = out-of-network. Likes and
dwells show on the row. The bar is hidden satisfaction — it drives whether
they open the app tomorrow.

I hit Tab: **default weights** vs an **engagement-tilted** config. Same
person, same 21 days. The trap feed looks busier on day 0. By week three
more cells go gray. That is a walk-back you can see before you ramp.

Around that I ran the actual search theories (not in the clip):

- Maximise the **2-day proxy** (likes + dwells) → you beat default this week
  and lose **retention**. Walk-back 5/5 in the pilot. The bidirectional-boost
  failure mode is the rule, not the exception.
- Spend the same budget on **delayed retention labels** → about **+0.10** vs
  doing nothing. The expensive metric is the product.
- **Fit a simulator** on those labels and extrapolate vs just picking the
  best measured config: confirmatory N=50, p=0.013, mean V **0.562 vs 0.560**.
  CI touches 0. A model of the labels does not replace buying more of them.
- A sim that **cannot see bait** is discarded by a live confirm — you ship
  default. You know it’s wrong when it’s blind to what makes people stop
  coming back.

**If you’re shipping a new feed from scratch:** don’t start with a
simulator. You have no delayed truth yet. Ship a conservative default → a
short delayed-retention A/B (default vs quality vs engagement) → only then
use a sim to *kill* bad candidates, never to crown them without a live
confirm.

**If you work on For You:** this does not say “change FavoriteWeight to
1.8.” Production is not this synthetic default. It does say: spend traffic
on delayed retention, not on a surrogate of the same labels; use offline
scores to stop 2-day-only wins before ramp; if replay can’t see emptiness,
it is not allowed to ship.

A simulator cannot replace Home Mixer experiments. It can stop the ones that
only win on two-day engagement — and you can watch that happen to one user,
one day at a time.

(Standalone synthetic benchmark, pre-registered. Not X production data.)

---

## Shorter hook (LinkedIn — use this with the video)

X For You doesn’t retrain the ranker every time the product changes. It
changes a **weight** on P(like), P(reply), P(report). `score = w · p̂`. The
open-source file even shows the human loop: a reply boost tried at 5 / 10 /
15 / 20, shipped at 20, walked back to 15.

I simulated that **outer** loop in a toy world — not live For You — then
drew **21 days of one user’s feed**. Video: same person, **default** vs
**max-engagement**. The busy feed on day 0 is the trap (rose = bait). The
gray cells by day 21 are people who didn’t come back.

What we actually learned, in three lines:

1. If you maximise the **2-day** metric, you will look like a win and
   reverse later. That’s the walk-back, on purpose.
2. Spend the same budget on **delayed retention** instead: about **+0.10**
   vs doing nothing. Waiting beats clever search on the wrong number.
3. A **fitted sim** of those delayed tests does not beat just running them
   (N=50: 0.562 vs 0.560, CI touches 0). Use a sim to **kill** bait-shaped
   configs, not to skip A/B. You know it’s lying when it can’t see what
   makes people stop coming back.

New feed? Default → a few delayed-retention tests → sim as a filter.
For You? Don’t paste these weights. Do spend traffic on the slow metric.

(Standalone synthetic benchmark. Not X production data.)

Longer write-up for DEV.to: [`21-devto.md`](21-devto.md).

---

## Honesty checklist before you hit Post

- [ ] Video is this pygame viewer, not a game of walking avatars
- [ ] You did **not** say you trained RL on live For You
- [ ] Weights on screen are synthetic B0 vs trap, not August `param.rs`
- [ ] “+0.10” is delayed-V vs default in the toy world, not X metrics
- [ ] M1 vs B5 is a **tiny** confirmatory edge, not “RL won”
