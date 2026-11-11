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

Shipped checkpoint (split seed 0, `python -m keyauth.train --data cmu`): val 13.56, test 21.84, threshold 0.7545,
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

Not as they are. The Embedder protocol (`Embedder` in src/keyauth/auth/engine.py) maps one attempt to one unit vector
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

C6b adds that scoring path's model half in src/keyauth/auth/scorers.py: `timing_vector` (one validated attempt ->
raw timing vector), `ScaledManhattanScorer` / `MahalanobisScorer` with `fit(enroll) -> ScorerState` and
`score(state, vector) -> float`, and JSON (de)serialization of the state so the store can persist it. Scores are
identical to keyauth.baselines. The engine wiring is not done yet.

## C6b: per-user thresholds

CMU, real data, 10 enrollment attempts, unseen users, split seeds 0 to 4 (the protocol above). Nothing here is
synthetic. **Every threshold and rule parameter was picked on val users only; test users only report.**

**Verdict: per-user thresholds do not measurably help.** Mean test EER moves by -0.26 points for the encoder, 0.00
for scaled Manhattan and -0.05 for Mahalanobis, far inside the split noise (std 5 to 10). On split seed 0 the shipped
encoder trades FRR for FAR: test FRR 36.76% -> 31.76%, FAR 13.93% -> 15.00%. The 37.79% FRR problem is mostly
a global offset (the val threshold is too strict for these 8 test users), and a rule built from enrollment
attempts cannot see that.

### The rule (src/keyauth/thresholds.py)

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

`python -m keyauth.train --data cmu` now also fits the rule on val users and writes it to the checkpoint and to
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
python -m keyauth.train --data cmu           # writes the shipped checkpoints/model.pt and model.json
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
as malformed). TEST_SECTION_ID is unique per presentation, so the sentence id is a hash of SENTENCE.

**Loader** (`load_aalto`, src/keyauth/data/aalto.py): each sentence goes through events.pair_events and
features.extract_features, like live capture (modifiers dropped, ordered by press time, hold <= 2 s, dd <= 5 s).
Defaults: drop sentences with any editing key (Backspace, Delete, arrows, Home/End, ...), like the login page rejects
corrected attempts; uncorrected typos are kept; 10 to 128 keystrokes; at least 5 clean sentences per participant.
Key codes only with `with_keys=True`.

**Statistics on the 10,000-participant subset** (`python -m keyauth.data.aalto data/aalto [options]`):

| | default (drop corrected, >= 5 sentences) | drop corrected, >= 10 sentences | keep corrections, >= 10 sentences |
|---|---|---|---|
| participants kept | 5,742 (57.4%) | 1,087 (10.9%) | 9,919 (99.2%) |
| sentences kept, of 150,000 | 43,439 (29.0%) | 12,241 (8.2%) | 145,686 (97.1%) |
| dropped: correction | 95,217 (63.5%) | 95,217 (63.5%) | 0 |
| dropped: timing outlier | 879 (0.6%) | 879 (0.6%) | 3,765 (2.5%) |
| dropped: too few sentences | 10,442 (7.0%) | 41,640 (27.8%) | 458 (0.3%) |
| dropped: malformed / unpaired / length | 13 / 10 / 0 | 13 / 10 / 0 | 13 / 30 / 48 |
| sentences per participant, mean (median) | 7.57 (7), range 5 to 15 | 11.26 (11), range 10 to 15 | 14.69 (15) |
| keystrokes per sentence, mean (median) | 37.3 (35), range 13 to 79 | 38.9 (37), range 14 to 75 | 46.6 (45), range 13 to 128 |
| distinct sentences | 1,525 | 1,522 | 1,525 |

Under the default, each sentence is typed by a median of 27 kept participants (5 to 61), and 24% of transitions have a
negative ud (rollover). Kept participants' keyboards (metadata): 3,056 laptop, 2,557 full, 79 small physical,
50 on-screen.

Caveat: dropping corrected sentences keeps the more accurate typists (participants with many corrections fall
under the 5-sentence minimum), so the kept population is not a random sample of the 10,000.
