# 14 — Stage 2 lock

Locked **2026-09-01**, before the confirmatory run, by executing the
pre-registered sample-size procedure on exploratory paired (M1 − B5)
normalised regret from seeds **10–14**. Those seeds are not confirmatory.

```bash
PYTHONPATH=src python -m rl_opt.stage2
```

Writes `analysis/stage2_lock.json`.

## N

Pilot sd of paired ΔR = **0.243** (dominated by seed 11, where M1 exploited
the surrogate). Paired-t n = 47. Wilcoxon inflation (ARE 3/π) → 50. Floor 20,
ceiling 200 → **N = 50**.

Confirmatory seeds: **30–79**. Development 20–24 and exploratory 10–14 are
never reused.

## Frozen world and budget

`WorldParams()` as of this lock, including `SAT_STEP=0.032`, `SAT_INIT=0.48`,
`DEFAULT_WEIGHTS = [1.8, 3.5, 0.55, 1.0, -43.2, -234.0]`. Budget **24,000**
user-days.

## L (realistic fidelity for H1)

Registered text said logs under `w₀`. The method that was tuned (once) and
frozen is the **shared 11-label delayed-V design** (100 users × 21 days,
`L = 11` labels, 23,100 user-days). Confirmatory H1 uses that, not a new
w₀-only log fitter. Logged as a deviation. Frozen M1 = `m1_gp_lcb_one`.

## What confirmatory H1 is

Wilcoxon signed-rank on paired (M1 − B5) in R, two-sided α = 0.05, median
ΔR with 10,000-resample bootstrap CI. MDE 0.10. Full N, no optional stopping.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.run_confirmatory
```

Checkpoints `analysis/confirmatory_h1.json` after every seed.

## Result

See [`17-h1.md`](17-h1.md). Wilcoxon p = 0.013, median ΔR = −0.025, bootstrap
CI [−0.038, 0]. Mean V: M1 0.562 vs B5 0.560. Statistically M1, practically a
tie next to the 0.10 gap over default.

