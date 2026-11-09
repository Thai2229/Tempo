# Plan

Status as of 2026-11-09. Each milestone = one session = one PR.

## Done
- [x] C1: scaffold, keystroke event pairing, timing features, synthetic typists
- [x] C2: data loaders, normalizer, user-disjoint splits, PyTorch dataset + PK sampler
- [x] C3: RhythmEncoder and triplet loss
- [x] C4: training, checkpoints, EER evaluation
- [x] C5: enrollment & verification engine (template store, adaptation, lockout, replay check)

Current model (checkpoints/model.json, CMU, 10 enrollment attempts): val EER 17.5%, test EER 28.5%,
early-stopped at epoch 2 of 12. Val and test disagree by 11 points, and val EER gets worse after epoch 2.

## C6: Model quality (owner: ai-lead)
- [ ] scripts/get_cmu.py: download DSL-StrongPasswordData.csv into data/ (source: CMU Keystroke Dynamics Benchmark page); fail clearly if unreachable
- [ ] Classic baselines on the same splits and enrollment count: scaled Manhattan and Mahalanobis distance on raw timing vectors
- [ ] Results table in docs/RESULTS.md: baseline vs encoder, val and test EER, mean and std over 3+ seeds / split seeds
- [ ] Investigate the val/test gap (split size, seed variance, early stopping on a noisy metric) and fix what's real
- [ ] Ship whichever model wins on test EER; engine keeps working with either via the Embedder protocol

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
