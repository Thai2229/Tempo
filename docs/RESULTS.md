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

## Known failure cases

- Split seed 3: encoder test EER 36.06%, the worst of the four methods on that split.
- At the val threshold on split seed 0, FRR is 37.79%: more than one genuine login in three is rejected.
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
```

JSON output goes to checkpoints/c6/ (gitignored). `--before-checkpoint` is optional and only adds the "Old checkpoint"
line; the pre-C6 file is gitignored and is not in the repo. Runs use 2 CPU threads per worker, so exact digits may
differ on other machines or thread counts. The soft-margin variant was removed from the code and from `dev`.
