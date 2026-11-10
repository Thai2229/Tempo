# Plan

Status as of 2026-11-09. Each milestone = one session = one PR.

## Done
- [x] C1: scaffold, keystroke event pairing, timing features, synthetic typists
- [x] C2: data loaders, normalizer, user-disjoint splits, PyTorch dataset + PK sampler
- [x] C3: RhythmEncoder and triplet loss
- [x] C4: training, checkpoints, EER evaluation
- [x] C5: enrollment & verification engine (template store, adaptation, lockout, replay check)

Current model (checkpoints/model.json, CMU, 10 enrollment attempts, split seed 0): val EER 13.56%, test EER 21.84%,
supervised contrastive loss, 30 epochs, no early stopping. Over 5 split seeds: val 18.76 +/- 4.47%, test 21.50 +/- 9.30%,
level with scaled Manhattan (22.14 +/- 5.98%). Details and caveats in docs/RESULTS.md.

## C6: Model quality (owner: ai-lead)
- [x] scripts/get_cmu.py: download DSL-StrongPasswordData.csv into data/ (source: CMU Keystroke Dynamics Benchmark page); fail clearly if unreachable
- [x] Classic baselines on the same splits and enrollment count: scaled Manhattan and Mahalanobis distance on raw timing vectors
- [x] Results table in docs/RESULTS.md: baseline vs encoder, val and test EER, mean and std over 3+ seeds / split seeds
- [x] Investigate the val/test gap (split size, seed variance, early stopping on a noisy metric) and fix what's real
- [x] Ship whichever model wins on test EER; engine keeps working with either via the Embedder protocol
  - Encoder shipped and runs through the Embedder protocol. The classic detectors tied it on test EER but need per-user statistics at scoring time, so they are not wired into the engine (docs/RESULTS.md; follow-up in C6b).

## C6b: Per-user verification (owner: ai-lead, backend-developer for engine wiring)
- [x] Per-user thresholds set at enrollment, to cut the false-reject rate (37.79% test FRR on split seed 0 at the global val threshold)
  - Opt-in (`KeyAuthEngine(per_user=...)`), default off: on CMU they trade FRR for FAR (seed 0 test FRR 36.76 -> 31.76, FAR 13.93 -> 15.00) and leave EER within noise (docs/RESULTS.md, C6b).
- [x] Optional support for the classic detectors in the engine (per-user statistics at scoring time)
  - `KeyAuthEngine(embedder=None, scorer=ScaledManhattanScorer() | MahalanobisScorer())`; state persisted as JSON in the store.

## C7: HTTP API (owner: backend-developer)
- [ ] FastAPI app in src/keyauth/api: POST /enroll, POST /verify, GET /status, DELETE /user, GET /health
- [ ] Request validation reuses auth/validation.py; consistent error shapes
- [ ] Rate limiting / lockout surfaced in responses; no raw timings or embeddings in logs
- [ ] Tests with FastAPI TestClient covering accept, reject, locked, replay, not enrolled

## C8: Web login demo (owner: frontend-developer, fullstack for wiring)
- [ ] Page that captures keydown/keyup with performance.now() for a fixed phrase and sends raw events
- [ ] Enrollment flow (10 samples with progress), then login with password + rhythm check
- [ ] Clear UI for accepted / rejected / locked, with the score shown in a debug mode
- [ ] Served by the API app; one command to run locally

## C9: Polish
- [ ] README: what it is, how it works, results table, demo GIF, how to run
- [ ] GitHub Actions CI running pytest
- [ ] Docker or one-click deploy option for the demo
