# C6 eval log

**Metric.** Global EER on unseen users (one threshold, as the engine uses it), CMU benchmark, 10 enrollment attempts,
impostors = first 5 attempts of every other user in the group. Lower is better. Per-user EER is reported alongside.

**Eval set.** No separate labelled set: the 51 CMU typists. Development used the 43 users outside split seed 0's test
set. `final` is the only run that scores test users, and it ran once.

**Keep rule.** A change stays only if it beats the previous variant on development holdout users.

Full tables and caveats: docs/RESULTS.md. Reproduce: `python scripts/compare_models.py noise|dev|stateless|final|table`.

| # | Run (one variable at a time) | Users | EER % | Decision |
|---|---|---|---|---|
| 0 | Baseline: batch-hard triplet, min-val early stopping | dev holdout, 3 splits | 37.26 +/- 7.70 | baseline |
| 1 | Classic detectors, no training (scaled Manhattan / Mahalanobis s=0.9) | same holdout | 24.50 / 23.69 | kept as baselines |
| 2 | lr 1e-4 | dev holdout | 35.48 +/- 9.53 | dropped: no clear gain, val worse |
| 3 | Soft-margin batch-hard (variant since removed) | dev holdout | 37.26 +/- 4.40 | dropped: no gain |
| 4 | Supervised contrastive, T=0.3 | dev holdout | 35.49 +/- 8.50 | dropped |
| 5 | Supervised contrastive, T=0.1 | dev holdout | 31.43 +/- 15.35 | kept: no collapse, best of the losses |
| 6 | T=0.1, EMA-smoothed stopping instead of min-val | dev holdout | 29.40 +/- 8.45 | dropped: no better than #7 |
| 7 | T=0.1, no early stopping (last of 30 epochs) | dev holdout | 28.91 +/- 11.16 | kept, shipped config |
| 8 | Final: before / after / scaled Manhattan / Mahalanobis | test, 5 split seeds | 28.28 / 21.50 / 22.14 / 23.27 | encoder shipped; tie with classics |

The encoder variants did not beat the classic detectors on development holdout users (28.91 against 24.50 and 23.69).
The final test run came out level instead; the encoder had 35 training users there rather than 27.
