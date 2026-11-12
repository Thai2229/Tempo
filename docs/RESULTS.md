# C6 results: classic baselines vs the rhythm encoder

All numbers are CMU Keystroke Dynamics Benchmark (real data, 51 typists), with **10 enrollment attempts** and
**unseen users**. Nothing on this page is synthetic. EER is the global EER (one threshold for everyone, which is
what the engine does) unless a column says per-user. Thresholds are picked on val users only.

## Summary

| | val EER | test EER | test minus val |
|---|---|---|---|
| Before: encoder as of C5 | 19.93 +/- 5.81 | 28.28 +/- 7.14 | +8.35 |
| Scaled Manhattan | 23.72 +/- 1.57 | 22.14 +/- 5.98 | -1.58 |
| Mahalanobis, shrinkage picked on val | 22.45 +/- 2.62 | 23.27 +/- 4.89 | +0.82 |
| **After: encoder as shipped** | 18.76 +/- 4.47 | **21.50 +/- 9.30** | +2.74 |

Mean +/- sample std over split seeds 0 to 4 (the same seed also seeds training), in percent.

**Shipped: the encoder** (supervised contrastive loss, 30 epochs, no early stopping), because it has the lowest
mean test EER, which is the rule in docs/PLAN.md. Read that result with these caveats:

- **It is a tie with the classic detectors, not a win.** Against scaled Manhattan the paired difference is -0.64
  points with a sample std of 5.69 (encoder better on 3 of 5 seeds). The improvement that holds up is against the
  encoder as it was: -6.78 points, better on 4 of 5 seeds.
- **Selecting the shipped model on test EER makes the winner's test number slightly optimistic.** The plan asks for
  that selection, and the same test users then report the number.
- The encoder is the least stable method across splits (test std 9.30, against 5.98 and 4.89; 36.06% on split seed 3).
- The classic detectors are better on per-user EER (18.58 and 18.88 against 20.53), which uses one threshold per user.
- Only the 8 test users of split seed 0 stayed out of development. The test users of seeds 1 to 4 are drawn from the
  same 51 people, so most of them were train or val users during development, and the five splits overlap each other.
  Treat the std as a rough indication, not independent evidence. On split seed 0 alone: encoder 21.84, scaled
  Manhattan 26.06, Mahalanobis 28.21, before 28.52.

## All four methods

**CMU, 10 enrollment attempts, split seeds [0, 1, 2, 3, 4] (mean +/- sample std, global EER, %)**

| method | val EER | test EER | mean test minus val | test per-user EER | FAR at val threshold | FRR at val threshold |
|---|---|---|---|---|---|---|
| encoder_after | 18.76 +/- 4.47 | 21.50 +/- 9.30 | +2.74 | 20.53 +/- 9.54 | 20.36 +/- 12.04 | 24.94 +/- 11.39 |
| encoder_before | 19.93 +/- 5.81 | 28.28 +/- 7.14 | +8.35 | 20.60 +/- 8.13 | 28.21 +/- 14.38 | 27.40 +/- 11.10 |
| mahalanobis | 22.45 +/- 2.62 | 23.27 +/- 4.89 | +0.82 | 18.58 +/- 2.77 | 25.36 +/- 10.15 | 21.20 +/- 7.04 |
| scaled_manhattan | 23.72 +/- 1.57 | 22.14 +/- 5.98 | -1.58 | 18.88 +/- 5.47 | 26.71 +/- 10.80 | 19.15 +/- 6.67 |

**Per seed (global EER, %)**

| method | seed | val | test | test minus val |
|---|---|---|---|---|
| encoder_after | 0 | 13.56 | 21.84 | +8.27 |
| encoder_after | 1 | 14.60 | 22.85 | +8.26 |
| encoder_after | 2 | 23.56 | 13.56 | -10.00 |
| encoder_after | 3 | 20.02 | 36.06 | +16.05 |
| encoder_after | 4 | 22.07 | 13.19 | -8.87 |
| encoder_before | 0 | 17.50 | 28.52 | +11.02 |
| encoder_before | 1 | 13.56 | 26.06 | +12.50 |
| encoder_before | 2 | 21.79 | 17.53 | -4.26 |
| encoder_before | 3 | 17.85 | 33.56 | +15.71 |
| encoder_before | 4 | 28.92 | 35.73 | +6.81 |
| mahalanobis | 0 | 25.79 | 28.21 | +2.42 |
| mahalanobis | 1 | 24.35 | 23.52 | -0.84 |
| mahalanobis | 2 | 19.57 | 20.71 | +1.14 |
| mahalanobis | 3 | 20.35 | 27.50 | +7.15 |
| mahalanobis | 4 | 22.21 | 16.44 | -5.77 |
| scaled_manhattan | 0 | 23.61 | 26.06 | +2.45 |
| scaled_manhattan | 1 | 23.21 | 24.65 | +1.44 |
| scaled_manhattan | 2 | 25.00 | 19.65 | -5.35 |
| scaled_manhattan | 3 | 21.44 | 27.50 | +6.06 |
| scaled_manhattan | 4 | 25.35 | 12.85 | -12.50 |

**encoder_after minus each method, paired by seed (test EER, points; negative = encoder better)**

| compared with | per seed | mean | sample std | seeds encoder wins |
|---|---|---|---|---|
| encoder_before | -6.68 -3.21 -3.97 +2.50 -22.53 | -6.78 | 9.42 | 4/5 |
| mahalanobis | -6.37 -0.66 -7.15 +8.56 -3.24 | -1.77 | 6.33 | 4/5 |
| scaled_manhattan | -4.23 -1.79 -6.08 +8.56 +0.34 | -0.64 | 5.69 | 3/5 |

Old checkpoint re-scored on split seed 0: val 17.50, test 28.52 (this confirms the "before" model).

Shipped checkpoint (split seed 0, `python -m tempo.train --data cmu`): val 13.56, test 21.84, threshold 0.7545,
at which test FAR is 14.29% and FRR is 37.79%. That imbalance shows the threshold does not transfer exactly between
small user groups.

## Why val and test disagreed

Development used only users outside split seed 0's test set: 27 train / 8 val / 8 holdout across 3 inner splits, plus
the original seed-0 val users. `final` was run once, after the training choices were frozen.

| Suspected cause | Verdict | Evidence | Change |
|---|---|---|---|
| Split size (8 val, 8 test users) | **Real, largest effect** | An 8-user EER has a std of 4.6 to 5.3 points; two random groups differ by 5 to 6 points on average, about 10 at p90, and by 11 or more in 6 to 8% of draws. No training is involved. Test minus val per seed ranges from -12.5 to +16.1. | Results now report mean and std over 5 split seeds. Split sizes are unchanged: more val users means fewer training users, and that trade-off was not tested. |
| Training-seed variance | **Real but small** | On a fixed split, val EER std across 5 training seeds is 1.0 to 1.7 points, against 4.5 to 5.8 across split seeds. | No code change; the multi-seed runs cover it. |
| Early stopping on a noisy metric | **Real** | The best-of-noisy-epochs val EER is biased low. Batch-hard holdout minus val is +13.93 with min-val stopping and +4.76 at the last epoch; contrastive is +8.44 and +4.15. Across 5 split seeds, mean test minus val fell from +8.35 to +2.74 (this also includes the loss change). | Default is now no early stopping (`patience=None`): train 30 epochs, keep the last. EMA-smoothed stopping was tried and was no better than the last epoch (holdout 29.40 against 28.91). |
| Embedding collapse (not on the original list) | **Real** | With batch-hard triplet loss the mean embedding has length 1.000 on the unit sphere, meaning all attempts sit almost on one point, and the old threshold was 0.028. Val EER rises with training on all 5 training seeds (best 19.21, last epoch 30.74), the same symptom as stopping at epoch 2. | Default loss is now supervised contrastive (T=0.1): mean-embedding length 0.715, last-epoch val EER 14.27 on the same split. |

A gap remains after these changes because 8 users is few. Treat any single-split EER as +/- 5 points.

**EER noise of a 8-user group (43 development users, 200 draws)**

| detector | mean EER | std | min | max | mean gap between two groups | p90 gap | P(gap >= 11 pts) | EER on all 43 |
|---|---|---|---|---|---|---|---|---|
| scaled_manhattan | 23.07 | 5.31 | 11.44 | 35.76 | 5.93 | 9.92 | 8.0% | 23.53 |
| mahalanobis_s0.9 | 22.04 | 4.64 | 11.48 | 34.60 | 5.41 | 10.00 | 6.0% | 22.18 |

**Encoder variants (development users only, 27 train / 8 val / 8 holdout, 3 inner splits, full 30 epochs)**

| variant | stop rule | val EER | holdout EER | holdout minus val | mean epoch kept | resultant length |
|---|---|---|---|---|---|---|
| batch_hard | raw min, patience 10 | 23.33 +/- 8.57 | 37.26 +/- 7.70 | +13.93 | 6.7 | 1.000 |
| batch_hard | EMA min, patience 10 | 25.50 +/- 10.37 | 37.49 +/- 2.37 | +11.99 | 9.3 | 1.000 |
| batch_hard | last epoch | 32.16 +/- 6.60 | 36.91 +/- 2.52 | +4.76 | 30.0 | 1.000 |
| batch_hard_lr1e-4 | raw min, patience 10 | 28.92 +/- 13.27 | 35.48 +/- 9.53 | +6.56 | 18.3 | 0.999 |
| batch_hard_lr1e-4 | EMA min, patience 10 | 28.95 +/- 13.27 | 35.84 +/- 9.51 | +6.90 | 26.3 | 0.999 |
| batch_hard_lr1e-4 | last epoch | 29.06 +/- 13.10 | 35.96 +/- 9.71 | +6.90 | 30.0 | 0.999 |
| supcon_t0.1 | raw min, patience 10 | 22.99 +/- 8.16 | 31.43 +/- 15.35 | +8.44 | 5.0 | 0.715 |
| supcon_t0.1 | EMA min, patience 10 | 23.46 +/- 8.94 | 29.40 +/- 8.45 | +5.94 | 18.0 | 0.715 |
| supcon_t0.1 | last epoch | 24.77 +/- 9.46 | 28.91 +/- 11.16 | +4.15 | 30.0 | 0.715 |
| supcon_t0.3 | raw min, patience 10 | 22.72 +/- 3.84 | 35.49 +/- 8.50 | +12.77 | 11.0 | 0.411 |
| supcon_t0.3 | EMA min, patience 10 | 23.91 +/- 4.95 | 37.74 +/- 8.03 | +13.83 | 17.0 | 0.411 |
| supcon_t0.3 | last epoch | 26.40 +/- 6.09 | 32.84 +/- 10.41 | +6.44 | 30.0 | 0.411 |

**Paired per inner split (same val and holdout users, encoder at its last epoch, EER %)**

| method | inner 1 val / holdout | inner 2 val / holdout | inner 3 val / holdout | mean val / holdout |
|---|---|---|---|---|
| scaled_manhattan | 27.2 / 20.0 | 33.1 / 22.5 | 11.3 / 31.0 | 23.88 / 24.50 |
| mahalanobis_s0.9 | 27.5 / 19.6 | 26.1 / 23.9 | 9.0 / 27.5 | 20.87 / 23.69 |
| batch_hard | 26.4 / 37.1 | 39.3 / 34.3 | 30.8 / 39.3 | 32.16 / 36.91 |
| batch_hard_lr1e-4 | 27.5 / 29.6 | 42.9 / 31.1 | 16.8 / 47.1 | 29.06 / 35.96 |
| supcon_t0.1 | 21.8 / 22.1 | 35.4 / 22.8 | 17.1 / 41.8 | 24.77 / 28.91 |
| supcon_t0.3 | 24.2 / 25.0 | 33.3 / 28.9 | 21.7 / 44.6 | 26.40 / 32.84 |

**Training-seed variance on the fixed seed-0 split (val users only)**

| variant | seeds | best val EER | final-epoch val EER |
|---|---|---|---|
| batch_hard@seed0split | 5 | 19.21 +/- 1.68 | 30.74 +/- 1.62 |
| supcon_t0.1@seed0split | 5 | 12.54 +/- 0.99 | 14.27 +/- 1.02 |

Holdout means come from only 3 inner splits, so differences of a few points between encoder variants are within
noise. Soft-margin batch-hard was also tried and dropped: holdout 36 to 37%, no better than batch-hard.

## Can the classic detectors run in the engine?

Not as they are. The Embedder protocol (`Embedder` in src/tempo/auth/engine.py) maps one attempt to one unit vector
with no knowledge of the user, and the engine scores L2 distance to a mean template. Scaled Manhattan and Mahalanobis
need per-user spread statistics at scoring time. Forcing classic features through a stateless embedding loses most of
their accuracy:

**Classic features through the stateless Embedder protocol (development users only, global EER %)**

| embedding | seed-0 val (8 users) | all 43 development users |
|---|---|---|
| per-user scaled Manhattan (reference, not an Embedder) | 23.61 | 23.53 |
| channel-normalized timings | 34.31 | 41.67 |
| log timings, speed-normalized | 47.45 | 48.40 |
| raw seconds | 42.16 | 44.24 |

The encoder shipped, so the engine needed no change. Preferring a classic detector later would need a per-user
scoring path in the engine, which is a design decision left open (see docs/PLAN.md).

C6b adds that scoring path's model half in src/tempo/auth/scorers.py: `timing_vector` (one validated attempt ->
raw timing vector), `ScaledManhattanScorer` / `MahalanobisScorer` with `fit(enroll) -> ScorerState` and
`score(state, vector) -> float`, and JSON (de)serialization of the state so the store can persist it. Scores are
identical to tempo.baselines. The engine wiring is not done yet.

## C6b: per-user thresholds

CMU, real data, 10 enrollment attempts, unseen users, split seeds 0 to 4 (the protocol above). Nothing here is
synthetic. **Every threshold and rule parameter was picked on val users only; test users only report.**

**Verdict: per-user thresholds do not measurably help.** Mean test EER moves by -0.26 points for the encoder, 0.00
for scaled Manhattan and -0.05 for Mahalanobis, far inside the split noise (std 5 to 10). On split seed 0 the shipped
encoder trades FRR for FAR: test FRR 36.76% -> 31.76%, FAR 13.93% -> 15.00%. The 37.79% FRR problem is mostly
a global offset (the val threshold is too strict for these 8 test users), and a rule built from enrollment
attempts cannot see that.

### The rule (src/tempo/thresholds.py)

For one user, from their own N = 10 enrollment attempts only:

```
loo_i       = score of enrollment attempt i against a model fitted on the other 9  (leave-one-out)
spread_u    = median_i loo_i
ref_u       = clip(weight * spread_u + (1 - weight) * reference_spread, 0.5 * reference_spread, 2 * reference_spread)
threshold_u = scale * ref_u            accept if score <= threshold_u
```

The same rule works for any "lower = more genuine" score: encoder distance to the template, scaled Manhattan,
Mahalanobis. Only the three global numbers change per scorer. They are fitted on val users only:

- `reference_spread` = median `spread_u` over val users.
- `weight` from the grid {0, 0.25, 0.5, 0.75, 1}, the one with the lowest val EER (ties go to the smaller weight).
  `weight = 0` is exactly the old global threshold, so the grid contains the baseline.
- `scale` = the EER threshold of the val scores divided by `ref_u`.
- The clip ratios 0.5 and 2 are fixed, not tuned. They are a safety bound so one sloppy enrollment cannot loosen a
  threshold without limit. They bound only 1 val user on 1 of 15 (method, seed) runs.

The EER "after" is the global EER of `score / ref_u`, which is the EER that per-user thresholds of this form can reach.
FAR and FRR "after" are at the val-fitted `scale`.

### Before / after

**Split seeds 0 to 4 (mean +/- sample std, %)**

| method | thresholds | val EER | val FAR | val FRR | test EER | test FAR | test FRR |
|---|---|---|---|---|---|---|---|
| encoder | global (before) | 18.86 +/- 4.68 | 18.86 +/- 4.70 | 18.87 +/- 4.65 | 21.50 +/- 9.52 | 20.36 +/- 11.58 | 24.64 +/- 11.16 |
| encoder | per-user (after) | 18.06 +/- 4.69 | 18.07 +/- 4.72 | 18.05 +/- 4.67 | 21.24 +/- 10.20 | 20.79 +/- 13.25 | 24.01 +/- 8.93 |
| scaled_manhattan | global (before) | 23.72 +/- 1.57 | 23.71 +/- 1.57 | 23.73 +/- 1.56 | 22.14 +/- 5.98 | 26.71 +/- 10.80 | 19.15 +/- 6.67 |
| scaled_manhattan | per-user (after) | 23.16 +/- 2.24 | 23.14 +/- 2.25 | 23.19 +/- 2.23 | 22.14 +/- 5.98 | 26.71 +/- 9.89 | 18.13 +/- 5.14 |
| mahalanobis | global (before) | 22.45 +/- 2.62 | 22.43 +/- 2.57 | 22.48 +/- 2.68 | 23.27 +/- 4.89 | 25.36 +/- 10.15 | 21.20 +/- 7.04 |
| mahalanobis | per-user (after) | 22.08 +/- 2.61 | 22.07 +/- 2.57 | 22.10 +/- 2.66 | 23.22 +/- 4.98 | 26.07 +/- 9.20 | 20.01 +/- 6.84 |

**Split seed 0 only (%)**

| method | thresholds | val EER | val FAR | val FRR | test EER | test FAR | test FRR |
|---|---|---|---|---|---|---|---|
| encoder | global (before) | 13.29 | 13.21 | 13.37 | 21.79 | 13.93 | 36.76 |
| encoder | per-user (after) | 13.21 | 13.21 | 13.21 | 21.42 | 15.00 | 31.76 |
| scaled_manhattan | global (before) | 23.61 | 23.57 | 23.65 | 26.06 | 28.93 | 23.46 |
| scaled_manhattan | per-user (after) | 23.61 | 23.57 | 23.65 | 26.06 | 28.93 | 23.46 |
| mahalanobis | global (before) | 25.79 | 25.71 | 25.87 | 28.21 | 29.29 | 25.38 |
| mahalanobis | per-user (after) | 25.79 | 25.71 | 25.87 | 28.21 | 29.29 | 25.38 |

The detector rows on seed 0 are identical because val picked `weight = 0`, which is the global threshold.

**Rule chosen on val, per seed**

| method | seed | weight | scale | reference spread | global-equivalent threshold | users clipped val / test | test FAR before -> after | test FRR before -> after |
|---|---|---|---|---|---|---|---|---|
| encoder | 0 | 0.25 | 1.5727 | 0.4822 | 0.7584 | 0 / 0 | 13.93 -> 15.00 | 36.76 -> 31.76 |
| encoder | 1 | 0.00 | 1.5929 | 0.5062 | 0.8064 | 0 / 0 | 17.14 -> 17.14 | 29.68 -> 29.68 |
| encoder | 2 | 0.00 | 1.6270 | 0.5259 | 0.8556 | 0 / 0 | 17.50 -> 17.50 | 10.16 -> 10.16 |
| encoder | 3 | 0.75 | 1.6957 | 0.4563 | 0.7737 | 1 / 0 | 40.71 -> 43.93 | 30.80 -> 28.40 |
| encoder | 4 | 0.25 | 1.3724 | 0.5401 | 0.7412 | 0 / 0 | 12.50 -> 10.36 | 15.80 -> 20.06 |
| scaled_manhattan | 0 | 0.00 | 1.7495 | 36.1865 | 63.3073 | 0 / 0 | 28.93 -> 28.93 | 23.46 -> 23.46 |
| scaled_manhattan | 1 | 1.00 | 1.6096 | 36.4986 | 58.7488 | 0 / 0 | 23.21 -> 25.71 | 28.27 -> 23.17 |
| scaled_manhattan | 2 | 0.00 | 1.7701 | 33.8596 | 59.9340 | 0 / 0 | 29.64 -> 29.64 | 12.18 -> 12.18 |
| scaled_manhattan | 3 | 0.75 | 1.7830 | 37.0128 | 65.9931 | 0 / 0 | 40.71 -> 38.21 | 17.76 -> 17.76 |
| scaled_manhattan | 4 | 0.00 | 1.7791 | 34.8641 | 62.0271 | 0 / 0 | 11.07 -> 11.07 | 14.10 -> 14.10 |
| mahalanobis | 0 | 0.00 | 1.6976 | 6.4630 | 10.9719 | 0 / 0 | 29.29 -> 29.29 | 25.38 -> 25.38 |
| mahalanobis | 1 | 1.00 | 1.6215 | 6.5378 | 10.6009 | 0 / 0 | 20.36 -> 27.50 | 26.51 -> 17.92 |
| mahalanobis | 2 | 0.00 | 1.9134 | 12.6490 | 24.2026 | 0 / 0 | 32.86 -> 32.86 | 9.01 -> 9.01 |
| mahalanobis | 3 | 1.00 | 1.7144 | 7.2123 | 12.3651 | 0 / 0 | 34.29 -> 30.71 | 22.44 -> 25.06 |
| mahalanobis | 4 | 0.00 | 1.6923 | 14.2104 | 24.0481 | 0 / 0 | 10.00 -> 10.00 | 22.66 -> 22.66 |

Mahalanobis shrinkage is picked on val per seed, as in `final`: 0.9, 1.0, 0.1, 0.7, 0.1 for seeds 0 to 4. The
weight val picks jumps between 0 and 1 from seed to seed, another sign that there is little signal to fit.

**Fixed weight, read only: nothing was chosen from this table (seeds 0 to 4, %)**

| method | weight | val EER | test EER | test FAR | test FRR |
|---|---|---|---|---|---|
| encoder | 0.00 | 18.86 +/- 4.68 | 21.50 +/- 9.52 | 20.36 +/- 11.58 | 24.64 +/- 11.16 |
| encoder | 0.25 | 18.42 +/- 4.62 | 20.86 +/- 9.36 | 19.64 +/- 12.96 | 24.15 +/- 7.61 |
| encoder | 0.50 | 18.75 +/- 4.99 | 21.37 +/- 9.22 | 19.86 +/- 14.02 | 25.29 +/- 4.37 |
| encoder | 0.75 | 19.65 +/- 5.27 | 23.00 +/- 9.46 | 21.43 +/- 15.08 | 27.92 +/- 5.29 |
| encoder | 1.00 | 22.13 +/- 6.08 | 25.27 +/- 9.56 | 24.00 +/- 17.76 | 31.11 +/- 10.45 |
| scaled_manhattan | 0.00 | 23.72 +/- 1.57 | 22.14 +/- 5.98 | 26.71 +/- 10.80 | 19.15 +/- 6.67 |
| scaled_manhattan | 0.25 | 23.94 +/- 2.08 | 22.17 +/- 6.05 | 27.00 +/- 10.66 | 19.21 +/- 6.46 |
| scaled_manhattan | 0.50 | 24.01 +/- 2.20 | 22.35 +/- 5.97 | 26.71 +/- 10.04 | 19.01 +/- 6.49 |
| scaled_manhattan | 0.75 | 23.79 +/- 2.80 | 22.56 +/- 5.61 | 27.07 +/- 9.90 | 18.54 +/- 6.45 |
| scaled_manhattan | 1.00 | 23.89 +/- 3.03 | 22.80 +/- 5.55 | 27.79 +/- 10.22 | 18.69 +/- 6.53 |
| mahalanobis | 0.00 | 22.45 +/- 2.62 | 23.27 +/- 4.89 | 25.36 +/- 10.15 | 21.20 +/- 7.04 |
| mahalanobis | 0.25 | 22.77 +/- 2.93 | 23.28 +/- 4.95 | 26.29 +/- 9.10 | 20.82 +/- 6.99 |
| mahalanobis | 0.50 | 23.66 +/- 3.06 | 23.50 +/- 4.54 | 26.79 +/- 8.41 | 20.85 +/- 7.01 |
| mahalanobis | 0.75 | 24.40 +/- 3.01 | 23.51 +/- 4.31 | 27.21 +/- 7.39 | 20.76 +/- 7.22 |
| mahalanobis | 1.00 | 24.63 +/- 3.77 | 23.79 +/- 3.93 | 28.07 +/- 7.61 | 20.37 +/- 7.67 |

A fully proportional threshold (weight 1) is clearly worse for the encoder (test FRR 31.11 against 24.64).

### Why it does not help

- **The classic detectors already normalize by the user's own spread.** Scaled Manhattan divides by each feature's
  mean absolute deviation and Mahalanobis by the covariance, so a loose typist does not get larger scores. On the 43
  development users (seed-0 test users excluded), the Spearman correlation between `spread_u` and the user's median
  genuine score is -0.03 (scaled Manhattan) and -0.04 (Mahalanobis, shrinkage 0.9). The rule has nothing to correct.
- **For the encoder the spread does predict genuine distances** (Spearman 0.88 on the 8 seed-0 val users, shipped
  checkpoint), but impostor distances to a loose user's template are also larger, so loosening that user's threshold
  admits impostors at about the rate it saves genuine attempts.
- **Drift is not visible at enrollment.** Enrollment is the first 10 attempts of session 1, and genuine attempts span
  8 sessions. On development users the median genuine score is about 1.3 to 1.4 times the enrollment spread over all
  sessions, but 1.0 to 1.1 within session 1.

### Shipped checkpoint

`python -m tempo.train --data cmu` now also fits the rule on val users and writes it to the checkpoint and to
model.json under `"per_user_threshold"`: weight 0.25, scale 1.5727, reference spread 0.4822 (global-equivalent
threshold 0.7584, close to the global val threshold 0.7568). Checkpoints without that key still load
(`PerUserThresholdParams.from_meta` returns None, and the engine keeps its global threshold).

This run, on a 4-core container: val 13.29, test 21.79, threshold 0.7568, test FAR 13.93 / FRR 36.76. The C6
numbers above (val 13.56, test 21.84, threshold 0.7545, FAR 14.29 / FRR 37.79) came from another machine. The small
differences come from CPU thread count. The `peruser` encoder runs are separate trainings with 2 threads per
worker, so their val EERs differ slightly from the `final` table (mean 18.86 against 18.76).

### Cost

No paid API. On CPU, setting one user's threshold at enrollment takes about 0.2 ms for the encoder (10 template
means; embeddings already computed), 0.4 ms for scaled Manhattan and 2.5 ms for Mahalanobis (10 fits with a 31x31
pseudo-inverse). Scoring one login with a stored detector state takes about 0.01 ms. A Mahalanobis state is about
20 KB as JSON (31x31 precision matrix in float64).

## Known failure cases

- Split seed 3: encoder test EER 36.06%, the worst of the four methods on that split.
- At the val threshold on split seed 0, FRR is 37.79%: more than one genuine login in three is rejected. Per-user
  thresholds (C6b) bring it to 31.76% at a FAR of 15.00% in this container's run, a trade rather than a fix.
- 10 enrollment attempts is short. Killourhy and Maxion (2009) enrolled with 200 attempts and report lower EER.
- Run through the engine on val users, validation refused 1 of 3,400 attempts (a CMU timing above the sanity bounds).
  The EER tables include such attempts.

## Cost per call

No paid API is involved. A verify call took 2 to 3 ms (median, two runs) on this laptop's CPU, in the engine, with
the shipped model. In that run, on seed-0 val users at the val threshold, 2696 of 3119 genuine attempts (86.4%) and 38
of 280 impostor attempts (13.6%) were accepted. This machine has no GPU; all training ran on CPU at about 2 to 3 s per epoch.

## Reproduce

All Python, on the CMU file (`python scripts/get_cmu.py` fetches it):

```
python scripts/compare_models.py noise       # 8-user sampling noise, development users only
python scripts/compare_models.py dev         # encoder variants, development users only
python scripts/compare_models.py stateless   # classic features through a stateless embedding
python scripts/compare_models.py final --before-checkpoint PATH/TO/OLD/model.pt   # reads test users
python scripts/compare_models.py table       # prints every table on this page
python -m tempo.train --data cmu           # writes the shipped checkpoints/model.pt and model.json
python scripts/compare_models.py engine      # shipped checkpoint through the engine, val users (latency, refusals)
python scripts/compare_models.py peruser     # C6b per-user thresholds, rule fitted on val, reads test users
```

JSON output goes to checkpoints/c6/ (gitignored). `--before-checkpoint` is optional and only adds the "Old checkpoint"
line; the pre-C6 file is gitignored and is not in the repo. Runs use 2 CPU threads per worker, so exact digits may
differ on other machines or thread counts. The soft-margin variant was removed from the code and from `dev`.

## C10: Aalto data

Dataset statistics only; no model results yet (that is C11).

**Source.** "136M Keystrokes": Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, Antti Oulasvirta. Observations on
Typing from 136 Million Keystrokes. CHI 2018. https://doi.org/10.1145/3173574.3174220,
https://userinterfaces.aalto.fi/136Mkeystrokes/. 168,593 participants of an online typing test, 15 transcribed
sentences each, drawn from a pool of 1,525 sentences.

**License (readme.txt):** "You are free to use this data for non-commercial use in your own research or projects with
attribution to the authors." The data stays in data/aalto/ (gitignored) and is never committed.

**Subset.** `python scripts/get_aalto.py` (10,000 participants, seed 0) reads the zip's central directory with HTTP
range requests and fetches only readme.txt, metadata_participants.txt and the chosen participant files: 150 MB
downloaded (20 MB central directory, 7 MB metadata, about 10 KB per participant), 10,012 range requests, 21 minutes
with 4 parallel requests, 1.14 GB on disk. The full archive is 1.5 GB zipped and 16 GB unzipped.

**Format found.** files/<id>_keystrokes.txt, tab separated, UTF-8, columns PARTICIPANT_ID, TEST_SECTION_ID,
SENTENCE, USER_INPUT, KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE. One row per key press with its release
already paired, times in Unix epoch milliseconds. LETTER is often empty or a name (SHIFT, BKSP, CAPS_LOCK, ARW_LEFT,
...), so key identity comes from KEYCODE (JavaScript keyCode). The final Enter / Next is not logged. A LETTER can be a
literal newline (the row spans two lines; the loader re-joins it) or a literal tab (10 fields; the sentence is dropped
as malformed). TEST_SECTION_ID is unique per presentation, so the sentence id is a hash of SENTENCE. metadata_participants.txt
has one row per participant, including KEYBOARD_TYPE (laptop / full / small / on-screen) and LAYOUT.

**Loader** (`load_aalto`, src/tempo/data/aalto.py): each sentence goes through events.pair_events and
features.extract_features, like live capture (modifiers dropped, ordered by press time, hold <= 2 s, dd <= 5 s).
Defaults (owner decisions, C10): keep corrected sentences, with the editing keys (Backspace, Delete, arrows,
Home/End, ...) as ordinary keystrokes, because people make mistakes typing passwords too; each sentence carries
`has_correction` so C11 can report EER on clean and corrected attempts separately. Only participants whose
KEYBOARD_TYPE is laptop or full (small physical and on-screen keyboards excluded; participants missing from the
metadata dropped). Uncorrected typos are kept; 10 to 128 keystrokes; at least 10 kept sentences per participant.
Key codes only with `with_keys=True`. `corrections="drop"` and `keyboards=None` give the other columns below.

**Statistics on the 10,000-participant subset** (`python -m tempo.data.aalto data/aalto [options]`):

| | default (keep corrections, >= 10 sentences, laptop + full) | drop corrected, >= 5 sentences, all keyboards | drop corrected, >= 10 sentences, all keyboards | keep corrections, >= 10 sentences, all keyboards |
|---|---|---|---|---|
| participants kept | 9,684 (96.8%) | 5,742 (57.4%) | 1,087 (10.9%) | 9,919 (99.2%) |
| dropped participants: small / on-screen keyboard; not in metadata | 239; 0 | - | - | - |
| sentences kept, of 150,000 | 142,248 (94.8%) | 43,439 (29.0%) | 12,241 (8.2%) | 145,686 (97.1%) |
| kept sentences with a correction | 89,765 (63.1%) | 0 | 0 | 91,968 (63.1%) |
| dropped: correction | 0 | 95,217 (63.5%) | 95,217 (63.5%) | 0 |
| dropped: timing outlier | 3,639 (2.4%) | 879 (0.6%) | 879 (0.6%) | 3,765 (2.5%) |
| dropped: too few sentences | 438 (0.3%) | 10,442 (7.0%) | 41,640 (27.8%) | 458 (0.3%) |
| dropped: malformed / unpaired / length | 13 / 30 / 47 | 13 / 10 / 0 | 13 / 10 / 0 | 13 / 30 / 48 |
| sentences per participant, mean (median) | 14.69 (15), range 10 to 15 | 7.57 (7), range 5 to 15 | 11.26 (11), range 10 to 15 | 14.69 (15) |
| keystrokes per sentence, mean (median) | 46.6 (45), range 13 to 128 | 37.3 (35), range 13 to 79 | 38.9 (37), range 14 to 75 | 46.6 (45), range 13 to 128 |
| distinct sentences | 1,525 | 1,525 | 1,522 | 1,525 |

Under the default: keyboards kept are 5,327 laptop and 4,357 full; each sentence is typed by a median of 93 kept
participants (59 to 130); clean sentences average 36.8 keystrokes and corrected ones 52.3. 9,339 participants have
at least one clean sentence (median 5 clean sentences each), so a clean-vs-corrected split is possible for almost
everyone. (The drop-corrected column: 24% of transitions have a negative ud, i.e. rollover.)

**Each participant typed each sentence only once** (checked on all 150,000 raw sentences, before any filtering: no
participant has the same SENTENCE twice). So a login-style test on Aalto ("enroll 10 attempts of one phrase, verify more
of the same phrase") is impossible; Aalto can only test unseen people on unseen sentences. The login-style test must
use CMU (51 people x 400 repetitions of .tie5Roanl).

Caveat: dropping corrected sentences (the comparison columns) keeps the more accurate typists, so that population is
not a random sample of the 10,000; the new default keeps 96.8% of participants and avoids this. The keyboard filter
removes 2.4% of participants.

## C11: Aalto-trained encoder, unseen phrases; CMU login test

Real data only: Aalto "136M Keystrokes" (subset from C10) and the CMU benchmark. Nothing here is synthetic. **Every
choice was made on val users only. Test users (Aalto test participants and each CMU split's test users) were scored
once, by `c11 final`, after the choices below were frozen.**

**Verdict: no improvement on the real test error, so the shipped model stays the C6 encoder.** Pretraining on 7,748
Aalto typists and then fine-tuning on CMU (B2) was picked on CMU val (18.72 against 19.10 for the current recipe).
On the same 15 test runs it scores 21.48 +/- 8.95, against 21.23 +/- 9.47 for the current recipe. On replicate 0,
the protocol behind the published 21.50, it scores 21.87 against 21.50. The Aalto model with no CMU training (B1) is
far worse on CMU (32.98). On Aalto itself the encoder works well on unseen people typing unseen sentences: 5.39 +/-
0.07% test EER.

### Data and splits

- **Aalto**: `load_aalto` defaults (corrections kept and flagged, >= 10 sentences, laptop / full keyboards): 9,684
  participants, 142,248 sentences, parsed once into checkpoints/c11/aalto.npz (gitignored, 42 s to parse).
  `split_phrase_independent` (seed 0): participants 80/10/10 (7,748 train / 968 val / 968 test), and 762 of the
  1,525 sentence ids held out. Training uses train participants' **seen** sentences only: 56,730 sentences (19,117 from
  5,258 participants when corrected sentences are dropped). Val and test use their participants' **held-out**
  sentences only, so every val / test attempt is an unseen person typing an unseen sentence. Participants need >= 6
  held-out sentences: 800 val participants (6,361 sentences) and 814 test participants (6,461 sentences).
- **Aalto protocol**: enroll the first **k = 5** held-out sentences of a participant (each sentence is typed once, so
  5 is what fits; about 7 held-out sentences per participant) -> template; genuine = the remaining 1 to 9 sentences
  (2,361 val / 2,391 test); impostors = the first 5 held-out sentences of every other participant in the same group
  (3.2M val / 3.3M test comparisons). Clean / corrected: genuine and impostor attempts filtered by `has_correction`,
  with the same templates (enrollment can mix both). Threshold = Aalto val EER threshold.
- **CMU protocol**: exactly C6's: split seeds 0 to 4 with `split_users` (35 train / 8 val / 8 test users), 10
  enrollment attempts, 5 impostor attempts per other user, one global threshold at the split's val EER, test EER and
  FAR / FRR at that threshold. Aalto-trained models get CMU **without its final Return** (Aalto logs no Enter); the
  CMU-only recipe keeps it, as shipped.
- **Replicates**: each CMU split is run 3 times (training seed = split seed + 100 * replicate); B1 / B2 replicate r
  uses the Aalto model trained with seed r. Replicate 0 of B3 is exactly C6's runs (it reproduces 18.86 / 21.50).

### Choices made on val

**Aalto variants, Aalto VAL participants (EER %, seed 0 unless stated; one change from `base` at a time, then the
combination of the winners)**

| variant | change from base | val EER | val clean | val corrected | decision |
|---|---|---|---|---|---|
| base | width 64, 1 GRU layer, T = 0.1, corrections kept, no key identity, 15 epochs | 6.61 | 6.85 | 6.45 | |
| drop_corrections | train on clean sentences only | 7.08 | 7.15 | 6.80 | keep corrections (owner's preference, and lower) |
| keys | key identity on (keyCode embedding) | 4.11 | 4.48 | 3.96 | not deployable (see below) |
| t0.05 | T = 0.05 | 6.23 | 6.42 | 6.02 | lower T |
| t0.025 | T = 0.025 | 6.15 | 6.41 | 5.95 | tie with 0.05 |
| deep | 2 GRU layers | 6.49 | 6.74 | 6.32 | tie with base at 1.6x the time; keep 1 layer |
| e30 | 30 epochs | 6.30 | 6.43 | 6.18 | 30 epochs |
| **t0.05_e30** (selected) | T = 0.05, 30 epochs | **5.90 +/- 0.03** (3 seeds) | 6.30 +/- 0.19 | 5.63 +/- 0.07 | selected |
| t0.05_e30_keys | the selected config with key identity | 3.18 | 3.40 | 3.04 | privacy cost measurement |

The binomial noise of one of these val EERs is about +/- 0.5 points (2,361 genuine attempts), so only "keys" and
"selected vs base" are clearly real. The combination uses T = 0.05 rather than 0.025 because it was started before the
0.025 run finished; the two tied (6.23 / 6.15). A width-128 variant was started and stopped after one epoch (7 min per
epoch on one core, 1.8 h per run), to stay in the compute budget, so only two sizes were compared. Training: supervised
contrastive, 64 participants x 4 sentences per batch, 200 batches per epoch, AdamW lr 1e-3, cosine schedule, last
epoch kept.

**CMU normalizer, B1, CMU VAL (split seeds 0-4, %)**: CMU train-user statistics beat the Aalto statistics for all 11
Aalto models (for the selected model, seeds 0 / 1 / 2: 28.58 / 28.92 / 29.58 against 29.63 / 29.50 / 33.22). Chosen:
**CMU train statistics** (`--normalizer fit`), for B1 and B2.

**B2 fine-tuning grid, CMU VAL (Aalto t0.05 model, replicate 0, split seeds 0-4, %)**

| lr \ epochs | 5 | 15 | 30 |
|---|---|---|---|
| 1e-4 | 24.14 +/- 4.98 | 21.28 +/- 3.60 | 20.80 +/- 4.47 |
| 3e-4 | 21.57 +/- 3.74 | 20.04 +/- 4.34 | 19.36 +/- 4.54 |
| 1e-3 | 20.52 +/- 5.18 | **19.07 +/- 4.34** | 19.65 +/- 4.54 |

Chosen: lr 1e-3, 15 epochs (supcon T = 0.1, the CMU recipe's batches). The grid ran on the round-1 model (t0.05)
while the final Aalto model was still training, and was not re-run. The base Aalto model matters little: fine-tuning
`base` / `drop_corrections` / `t0.05` / `t0.05_e30` (replicate 0) gives 20.07 / 19.63 / 19.07 / 18.48.

**Selection of the model to ship, deployable variants only, CMU VAL (mean over 5 splits x 3 replicates, %)**

| variant | val EER | |
|---|---|---|
| B1: Aalto t0.05_e30, no CMU training | 29.03 +/- 6.26 | |
| **B2: Aalto t0.05_e30 + CMU fine-tune** | **18.72 +/- 3.50** | **selected on val** |
| B3: CMU-only recipe (shipped) | 19.10 +/- 4.67 | |

The margin is 0.38 points; per split B2 is ahead on 3 of 5 (+1.41, +0.02, -1.45, -0.24, -1.66 for B2 minus B3).
B2's val number is also the minimum of a 9-cell grid, so it is biased low.

### Final table (test scored once)

**CMU login test, 10 enrollment attempts, split seeds 0-4: mean +/- std across the 5 splits of the
replicate-averaged values (%); FAR / FRR at each split's val threshold**

| variant | deployable | val EER | test EER | test FAR | test FRR | test EER std across 3 training seeds | test EER, replicate 0 |
|---|---|---|---|---|---|---|---|
| Current shipped encoder (C6, published) | yes | 18.76 +/- 4.47 | 21.50 +/- 9.30 | 20.36 +/- 12.04 | 24.94 +/- 11.39 | - | 21.50 |
| Scaled Manhattan (C6, published) | (scorer) | 23.72 +/- 1.57 | 22.14 +/- 5.98 | 26.71 +/- 10.80 | 19.15 +/- 6.67 | - | - |
| B3: CMU-only recipe, re-run (Return kept) | yes | 19.10 +/- 4.67 | **21.23 +/- 9.47** | 20.74 +/- 11.94 | 22.93 +/- 9.98 | 0.58 | 21.50 +/- 9.52 |
| **B2: Aalto t0.05_e30 + CMU fine-tune (selected on val)** | yes | 18.72 +/- 3.50 | 21.48 +/- 8.95 | 20.52 +/- 12.35 | 24.72 +/- 9.95 | 0.49 | 21.87 +/- 9.15 |
| B1: Aalto t0.05_e30, no CMU training | yes | 29.03 +/- 6.26 | 32.98 +/- 10.80 | 33.38 +/- 15.55 | 35.06 +/- 10.60 | 0.27 | 32.74 +/- 10.77 |
| B1: Aalto t0.05_e30_keys (key identity) | no | 29.07 +/- 6.25 | 34.27 +/- 12.81 | 33.36 +/- 15.97 | 36.13 +/- 13.21 | - | 34.27 |
| B1: Aalto keys (base config, key identity) | no | 27.27 +/- 5.81 | 32.21 +/- 13.70 | 32.71 +/- 17.38 | 32.31 +/- 14.75 | - | 32.21 |

The C6 encoder's published val (18.76) came from another machine; this container's identical runs give 18.86 (C6b
noted the same thread-count effect), and the same test 21.50. B1 rows for the other Aalto variants (seed 0, no CMU
training) are all 32.2 to 34.5 test EER; `python scripts/compare_models.py c11 table` prints them.

**Per split, test EER (%, replicate mean)**

| variant | split 0 | split 1 | split 2 | split 3 | split 4 |
|---|---|---|---|---|---|
| B3: CMU-only recipe | 22.61 | 22.38 | 13.23 | 35.72 | 12.22 |
| B2: Aalto + CMU fine-tune | 26.46 | 22.61 | 13.81 | 33.08 | 11.43 |
| B1: Aalto, no CMU training | 31.31 | 35.95 | 22.53 | 49.88 | 25.25 |

B2 minus B3, paired by split: +3.85, +0.23, +0.58, -2.64, -0.79 (mean +0.24, sample std 2.37; B2 better on 2 of 5).

**Aalto, unseen participants typing unseen sentences, 5 enrollment sentences (EER %; FAR / FRR at the Aalto val
threshold)**

| variant | seeds | val EER | test EER | test clean | test corrected | test FAR | test FRR |
|---|---|---|---|---|---|---|---|
| base | 1 | 6.61 | 6.02 | 6.16 | 5.94 | 6.17 | 5.77 |
| drop_corrections | 1 | 7.08 | 6.40 | 6.37 | 6.45 | 6.69 | 6.02 |
| t0.05 | 1 | 6.23 | 5.78 | 5.94 | 5.74 | 5.73 | 5.94 |
| t0.025 | 1 | 6.15 | 5.99 | 6.03 | 5.93 | 5.64 | 6.40 |
| deep | 1 | 6.49 | 5.91 | 6.18 | 5.74 | 6.12 | 5.65 |
| e30 | 1 | 6.30 | 5.77 | 5.94 | 5.68 | 5.90 | 5.40 |
| **t0.05_e30 (selected, keys off)** | 3 | 5.90 +/- 0.03 | **5.39 +/- 0.07** | 5.63 +/- 0.07 | 5.31 +/- 0.08 | 5.47 +/- 0.04 | 5.33 +/- 0.17 |
| keys (base config, keys on) | 1 | 4.11 | 4.06 | 4.22 | 3.95 | 3.84 | 4.35 |
| t0.05_e30_keys (keys on) | 1 | 3.18 | **3.01** | 3.41 | 2.76 | 2.90 | 3.18 |

Test: 814 participants, 6,461 sentences, 2,391 genuine (876 clean / 1,515 corrected) and 3.3M impostor comparisons.

### Clean vs corrected attempts

Corrected attempts are not harder: on test, the selected model's EER is 5.63 on clean and 5.31 on corrected attempts
(val: 6.30 / 5.63). Corrected sentences are longer (52 keystrokes on average against 37 in C10's statistics), and
longer text carries more rhythm. Training on clean sentences only was worse on both (val 7.15 / 6.80 against 6.85 /
6.45 for base), mostly because it leaves a third of the training data.

### Key identity (the privacy cost of anonymized events)

The web page sends anonymized keys, so only models without key identity can be deployed. That anonymization costs
real accuracy on free text: Aalto test EER 5.39 without key identity against 3.01 with it at the selected config
(val 5.90 against 3.18), and 6.02 against 4.06 at the base config. Key identity lets the model learn per-key and
per-digraph timing (and the keyboard layout). On CMU it makes no consistent difference (B1 test 34.27 with against
32.98 without at the selected config, 32.21 against 33.44 at the base config): everyone types the same 10 keys, so
the key sequence carries no information about the person. A deployed
phrase-independent service would face the same trade-off: key identity cuts the free-text error by about 45%, but the
server would then receive what the user typed.

### Shipped

**Kept: checkpoints/model.pt is unchanged (the C6 encoder, md5 27dbad32...).** B2 was selected on val, but on test it
does not beat the current model: 21.48 against 21.23 on the same 15 runs, 21.87 against 21.50 on replicate 0, and
26.46 against 22.61 on split seed 0, the split the shipped checkpoint is trained on. So nothing was written to
model.pt and no model_c6.pt copy was needed.

What exists for later use (all gitignored): `python -m tempo.train --data cmu --init checkpoints/c11/aalto_t0.05_e30_s0.pt
--epochs 15 --lr 1e-3 --out checkpoints/c11/model_b2_split0.pt` writes the B2 model for split seed 0 the way train.py
writes model.pt: weights, normalizer, val threshold, per-user threshold params (weight 0.25, scale 1.5624), and
`"final_enter": false`. It reproduces the split-0 replicate-0 run exactly (val 15.35, test 28.29). Loaded through
the API's `build_engine`, it enrolls and verifies CMU attempts that include the final Return: TorchEmbedder drops the
last keystroke because the checkpoint says `final_enter: false`, while the engine still stores and length-checks the
full attempt. On seed-0 val users through the engine (`engine` command, B2 split-0 checkpoint): 2,672 of 3,119 genuine
and 40 of 280 impostor attempts accepted, 1 attempt refused by validation, no enrollment failed.

### Code

- `python -m tempo.train --data aalto` (cache `--aalto-cache`, `--corrections keep|drop`, `--keys`, `--width`,
  `--layers`, `--temperature`, `--lr`, `--p`, `--k`; `--n-enroll` defaults to 5 for Aalto) and `--init CHECKPOINT`
  for fine-tuning (`--normalizer fit|init`; `--epochs 0` only calibrates the threshold). `--init` inherits
  `final_enter`, so CMU's Return is dropped when fine-tuning an Aalto model.
- Checkpoints record `final_enter` (missing = True, i.e. checkpoints from before C11). `TorchEmbedder` drops the last
  keystroke when it is False, refuses attempts with fewer than 3 keystrokes then (422 invalid_timings in the API), and
  refuses key-identity models, which the engine cannot feed.
- `EncoderConfig.key_vocab` (0 = off) adds a keyCode embedding; `KeystrokeData.drop_final_key`, `save_npz` /
  `load_npz`; `split_phrase_independent`; train / embed trim padding to each batch's longest attempt (no change for
  CMU, where every attempt has the same length).

### Compute

4-core CPU container, no GPU, 1 torch thread per job, up to 4 jobs at a time. About 3.5 hours of wall clock for the
whole milestone. Aalto training: 11 runs, 9.0 core-hours (35 min per 15-epoch run, 64 min per 30-epoch run; the
2-layer model 55 min), plus about 45 core-minutes in the stopped width-128 / round-2 attempt. CMU: B3 15 runs in 17
minutes, B2 70 fine-tuning runs in 27 core-minutes (0.1 to 0.7 min each), B1 and `final` a few minutes. No paid API.

### Caveats

- **Typing-test sentences are not passwords.** Aalto is transcription of English sentences, typed once each, in one
  sitting, on the participant's own keyboard. A password is short, practiced, and typed many times. This is the
  likely reason B1 transfers so badly (33% test EER): the Aalto model never saw a phrase typed twice.
- **8-user CMU splits are noisy.** One split's EER moves by about +/- 5 points between user groups (C6). Both the
  0.38-point val advantage of B2 and the 0.25-point test disadvantage are far inside that noise: B2 and B3 are a tie,
  and the training-seed std (about 0.5) is small next to the split std (about 9).
- **Selection bias.** B2's val number is the best of a 9-cell grid. B3's recipe was itself chosen in C6 on development
  users that overlap these splits (only seed 0's test users were held out then).
- **Aalto val noise.** About +/- 0.5 points per val EER (about 3 genuine attempts per participant). The ordering of
  the deployable variants other than the selected one is within that noise.
- **Return dropped.** Aalto-trained models do not use the Return keystroke, which the CMU-only encoder uses (its hold,
  and the latency from the last character into Return). That is a handicap for B1 / B2 on CMU, decided in C10. Whether
  fine-tuning with the Return kept would help was not measured here; C11b below measures it (it does not help).
- **License.** Aalto is free for non-commercial use with attribution. A model trained on it inherits that restriction,
  which is one more reason it is not shipped now.
- One Aalto data split (seed 0) and one sentence split; Aalto std is over training seeds only.

### Reproduce

```
python scripts/compare_models.py c11 aalto --variants base,drop_corrections,keys,t0.05 --workers 4
python scripts/compare_models.py c11 aalto --variants t0.025 --workers 1   # and deep, e30, t0.05_e30, t0.05_e30_keys
python scripts/compare_models.py c11 aalto --variants t0.05_e30 --seeds 1,2 --workers 2
python scripts/compare_models.py c11 b3 --workers 1
python scripts/compare_models.py c11 b1 --workers 4
python scripts/compare_models.py c11 b2 --b2-base t0.05 --workers 1        # lr / epochs grid
python scripts/compare_models.py c11 b2 --selected-only --replicates 0,1,2 # selected Aalto model, chosen lr / epochs
python scripts/compare_models.py c11 b2 --b2-base base --selected-only     # and drop_corrections (Aalto base check)
python scripts/compare_models.py c11 final --workers 4                     # reads test users; run once
python scripts/compare_models.py c11 table
```

Decisions are in `C11_SELECTED` in scripts/compare_models.py. JSON, logs and checkpoints go to checkpoints/c11/
(gitignored). The first Aalto command parses data/aalto into checkpoints/c11/aalto.npz.

## C11b: fine-tuning the Aalto model with CMU's Return kept

Real CMU data and the C11 Aalto checkpoints (not retrained). Same protocol as C11: split seeds 0-4, 3 replicates
each, 10 enrollment attempts, one global threshold at each split's val EER. **Choices were made on CMU val users
only; B2e's test users were scored once, by `c11 final_b2e`, after `C11B_SELECTED` was frozen.** B2 and B3 test
numbers are C11's (final.json), not re-scored.

**Question.** B2 (Aalto pretrain + CMU fine-tune) dropped CMU's final Return because the Aalto log has no Enter. B2e
fine-tunes the same Aalto models (t0.05_e30, seeds 0 / 1 / 2) with the Return kept, so the model also sees the
latency from `l` into Return and the Return's hold, as the CMU-only recipe (B3) does.

**Verdict: keeping the Return does not help, and nothing is shipped.** On val, B2e is worse than both B2 and B3
(19.84 against 18.72 and 19.10), and worse than B2 in all 9 cells of the same fine-tuning grid. On test it is 20.33
against 21.23 for B3, but that 0.91-point gap comes from one split (split 3: -5.58; the other four are 0.0 to +0.5
points worse). It is far inside the paired noise (95% interval +/- 3.25). The val and test orderings disagree, as
they did in C11. This is a tie.

### Input handling

No code change was needed for the encoder to accept the Return. Both datasets use the same convention: the last
keystroke has only a hold, and its dd / ud are masked (`valid_mask`) and zeroed. With the Return kept, CMU attempts
are 11 keystrokes instead of 10, the 10th key's latency into Return becomes a valid input, and the Return is the
hold-only last key. The Conv / packed-GRU encoder takes any length. The architecture is unchanged. The B2e
checkpoints record `final_enter: true`, so `TorchEmbedder` passes the full attempt through.

### Choices made on val

**Fine-tuning grid, the same 3 x 3 as C11 (CMU VAL, Aalto t0.05_e30 seed 0, replicate 0, CMU-train normalizer,
split seeds 0-4; EER %).** For a paired comparison on the same base model, B2 (Return dropped) was also run on this
grid. Its C11-selected cell (lr 1e-3, 15 epochs) reproduced C11's 18.48 bit for bit.

| lr | epochs | B2e (Return kept) | B2 (Return dropped) | B2e - B2 per split (points) | mean |
|---|---|---|---|---|---|
| 1e-4 | 5 | 26.97 +/- 6.39 | 23.81 +/- 4.75 | +0.32 +2.85 +6.85 +1.45 +4.29 | +3.15 |
| 1e-4 | 15 | 25.09 +/- 5.70 | 21.36 +/- 3.58 | -0.63 +3.19 +7.92 +1.71 +6.44 | +3.73 |
| 1e-4 | 30 | 23.78 +/- 5.18 | 20.67 +/- 4.13 | +2.15 +2.50 +7.44 -0.39 +3.87 | +3.11 |
| 3e-4 | 5 | 25.23 +/- 5.55 | 21.65 +/- 3.68 | -0.03 +3.23 +8.61 +0.35 +5.71 | +3.57 |
| 3e-4 | 15 | 22.63 +/- 5.55 | 20.00 +/- 3.72 | +0.66 +1.06 +6.00 +1.13 +4.27 | +2.63 |
| 3e-4 | 30 | 22.65 +/- 4.87 | 19.20 +/- 3.64 | +1.06 +3.26 +4.65 +3.60 +4.68 | +3.45 |
| 1e-3 | 5 | 22.63 +/- 5.94 | 19.79 +/- 4.63 | +1.37 +2.15 +4.65 +1.74 +4.29 | +2.84 |
| 1e-3 | 15 | 21.41 +/- 5.60 | 18.48 +/- 4.03 | +1.44 +1.08 +3.52 +3.23 +5.37 | +2.93 |
| **1e-3** | **30** | **20.23 +/- 5.47** | 18.36 +/- 4.37 | -0.35 +1.77 +1.10 +2.21 +4.65 | +1.87 |

Chosen for B2e: lr 1e-3, 30 epochs (the grid minimum). The gap to B2 shrinks with more training (+3.2 at the
gentlest cell, +1.9 at the strongest), which is what you would expect if the model has to unlearn the "no Enter"
input before the Return can help. B2 stays at its C11 choice (lr 1e-3, 15 epochs); it was not re-selected.

**Normalizer at the chosen cell (replicate 0, CMU VAL):** CMU-train statistics 20.23 +/- 5.47 against Aalto
statistics 21.58 +/- 6.76. Chosen: **CMU train statistics**, the same as B2. No layers were frozen (full fine-tune, as
in B2), and no other knob was tried, so the selection space (9 grid cells + 2 normalizers) is the same size as B2's.

**Selection among the deployable variants (CMU VAL, 5 splits x 3 replicates, %)**

| variant | val EER | |
|---|---|---|
| B3: CMU-only recipe (shipped recipe) | 19.10 +/- 4.67 | |
| B2: Aalto + CMU fine-tune, Return dropped | **18.72 +/- 3.50** | best on val (C11) |
| B2e: Aalto + CMU fine-tune, Return kept | 19.84 +/- 4.83 | not selected |

B2e minus B3 on val, per split: +1.56, +0.75, +0.62, -0.92, +1.68 (B2e better on 1 of 5). B2e minus B2: +0.15, +0.74,
+2.07, -0.69, +3.34. The ship rule was fixed before test was scored. Ship only if B2e is selected on val **and** its
paired test gain over B3 lies outside the 95% t interval of the 5 split differences. The first condition already
failed.

### Final table (B2e test scored once)

**CMU login test, 10 enrollment attempts, split seeds 0-4: mean +/- std across the 5 splits of the
replicate-averaged values (%); FAR / FRR at each split's val threshold**

| variant | runs | val EER | test EER | test FAR | test FRR | test EER std across 3 training seeds | test EER, replicate 0 |
|---|---|---|---|---|---|---|---|
| B3: CMU-only recipe (Return kept; C11) | 15 | 19.10 +/- 4.67 | 21.23 +/- 9.47 | 20.74 +/- 11.94 | 22.93 +/- 9.98 | 0.58 | 21.50 +/- 9.52 |
| B2: Aalto + CMU fine-tune, Return dropped (C11) | 15 | 18.72 +/- 3.50 | 21.48 +/- 8.95 | 20.52 +/- 12.35 | 24.72 +/- 9.95 | 0.49 | 21.87 +/- 9.15 |
| B2e: Aalto + CMU fine-tune, Return kept (lr 1e-3, 30 epochs) | 15 | 19.84 +/- 4.83 | 20.33 +/- 7.31 | 18.14 +/- 8.32 | 24.54 +/- 9.18 | 0.74 | 21.10 +/- 8.16 |

**Per split (test EER %, replicate mean)**

| variant | split 0 | split 1 | split 2 | split 3 | split 4 |
|---|---|---|---|---|---|
| B3 | 22.61 | 22.38 | 13.23 | 35.72 | 12.22 |
| B2 | 26.46 | 22.61 | 13.81 | 33.08 | 11.43 |
| B2e | 22.97 | 22.40 | 13.35 | 30.14 | 12.76 |

**Paired by split (points; negative = B2e better)**

| difference | per split | mean | sample std | 95% t half-width (4 df) | splits B2e better |
|---|---|---|---|---|---|
| B2e - B3, test EER | +0.37 +0.02 +0.12 -5.58 +0.54 | -0.91 | 2.62 | +/- 3.25 | 1/5 |
| B2e - B2, test EER | -3.48 -0.21 -0.45 -2.94 +1.33 | -1.15 | 2.01 | +/- 2.50 | 4/5 |
| B2e - B3, val EER | +1.56 +0.75 +0.62 -0.92 +1.68 | +0.74 | 1.04 | +/- 1.30 | 1/5 |
| B2e - B2, val EER | +0.15 +0.74 +2.07 -0.69 +3.34 | +1.12 | 1.60 | +/- 1.98 | 1/5 |

### Shipped

**Nothing. checkpoints/model.pt is unchanged (the C6 encoder, md5 27dbad32...).** Both conditions failed. B2e was not
selected on val (19.84, the worst of the three). Its 0.91-point test gain over B3 is inside the paired noise
(+/- 3.25) and comes from one split. An Aalto-derived model would also inherit Aalto's non-commercial license. If a
later milestone ships one anyway: `python -m tempo.train --data cmu --init checkpoints/c11/aalto_t0.05_e30_s0.pt
--keep-enter --epochs 30 --lr 1e-3` writes it with `"final_enter": true`, so the API embeds the full attempt.

### Caveats

- **Val and test disagree in sign.** On val, keeping the Return looks worse than B2 in every grid cell. On test, B2e
  is the lowest mean of the three. With 8 users per split, both are within the noise. The val result is the only one
  that may be used for decisions, and it says no.
- **Selection bias.** B2e's replicate-0 val number is the minimum of 9 grid cells on the same Aalto model that was
  then used. B2's grid ran on the round-1 model (t0.05), so B2's own selection was slightly less tied to its final
  runs. Both val numbers are biased low, B2e's a little more.
- **The chosen cell is at the grid edge** (highest lr, most epochs). A longer fine-tune might lower B2e further, but
  the trend points toward the CMU-only recipe, not past it. Extending the grid would add selection bias and was not
  done.
- B2 and B3 test numbers are C11's, scored once there. Only B2e's test users were read in C11b.

### Code and compute

- `python -m tempo.train --init CKPT --keep-enter`: fine-tune with the final Enter / Return kept, even if the init
  model was trained without it. The checkpoint gets `final_enter: true`. It is refused without `--init`, and with
  `--data aalto` (no Enter to keep).
- `scripts/compare_models.py c11 b2 --keep-enter [--normalizer aalto|cmu]` -> c11/b2e.json, checkpoints in
  c11/b2e/. `c11 final_b2e` scores B2e's test users once -> c11/final_b2e.json. `c11 table` prints the C11b tables.
- 4-core CPU, 1 torch thread per job, 4 jobs at a time. 105 fine-tuning runs (45 B2e grid, 45 B2 grid of which 5
  repeat C11 runs, 5 normalizer, 10 replicates) and 15 test scorings: about 13 minutes of wall clock and 41
  core-minutes. No paid API.

```
python scripts/compare_models.py c11 b2 --keep-enter --workers 4                        # grid, replicate 0
python scripts/compare_models.py c11 b2 --workers 4                                     # B2 on the same grid (paired)
python scripts/compare_models.py c11 b2 --keep-enter --selected-only --normalizer aalto # normalizer check
python scripts/compare_models.py c11 b2 --keep-enter --selected-only --replicates 1,2
python scripts/compare_models.py c11 final_b2e --workers 4                              # reads test users; run once
python scripts/compare_models.py c11 table
```
