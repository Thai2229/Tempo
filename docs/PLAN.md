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
  - `KeyAuthEngine(per_user=...)`; on by default in the API (`KEYAUTH_PER_USER=1`, owner decision in C7). On CMU they trade FRR for FAR (seed 0 test FRR 36.76 -> 31.76, FAR 13.93 -> 15.00) and leave EER within noise (docs/RESULTS.md, C6b).
- [x] Optional support for the classic detectors in the engine (per-user statistics at scoring time)
  - `KeyAuthEngine(embedder=None, scorer=ScaledManhattanScorer() | MahalanobisScorer())`; state persisted as JSON in the store.

## C7: HTTP API (owner: backend-developer)
- [x] FastAPI app in src/keyauth/api: POST /enroll, POST /verify, GET /status, DELETE /user, GET /health
- [x] Request validation reuses auth/validation.py; consistent error shapes
- [x] Rate limiting / lockout surfaced in responses; no raw timings or embeddings in logs
- [x] Tests with FastAPI TestClient covering accept, reject, locked, replay, not enrolled

## C8: Web login demo (owner: frontend-developer, fullstack for wiring)
- [x] Page that captures keydown/keyup with performance.now() for a fixed phrase and sends raw events
- [x] Enrollment flow (10 samples with progress), then login with password + rhythm check
- [x] Clear UI for accepted / rejected / locked, with the score shown in a debug mode
- [x] Served by the API app; one command to run locally

## C9: Polish
- [x] README: what it is, how it works, results table, demo GIF, how to run
- [x] GitHub Actions CI running pytest
- [x] Docker or one-click deploy option for the demo
  - Dockerfile; CI's `docker` job builds the image and smoke-tests it on every PR (the cloud session itself could not build it: download.pytorch.org blocked).

## Next phase (paused 2026-11-10; resume here)
Goal: a model that works on any password (C10-C11), then a Turnstile-style service other websites call (C12-C13).
Split agreed with the owner: Claude builds C12-C13 and the C10 loader/training scripts; the owner leads the C11
experiments and their interpretation.

Where we stopped: C10 done (data script, loader, sentence split; stats in docs/RESULTS.md). userinterfaces.aalto.fi is
now allowed in the "projects" cloud environment. Next is C11, led by the owner. Open questions for C11 are in the C10 PR
(correction policy, min sentences per participant, on-screen keyboards, CMU's trailing Enter). License: non-commercial
only, so C12-C13 stay a portfolio demo unless the authors grant permission.

## C10: Phrase-independent data (owner: ai-lead)
- [x] Owner reads the Aalto "136M Keystrokes" dataset license/terms (Dhakal et al., CHI 2018) before any download
  - "free to use for non-commercial use in your own research or projects with attribution"; nothing on redistribution, so the data is never committed or re-hosted.
- [x] scripts/get_aalto.py: download a subset (~10k participants) into data/, validated, fail clearly if unreachable
- [x] Loader: per-participant press/release times -> (hold, dd, ud) timings, user-disjoint splits; never commit the data
- [x] Keep key identity optional in the loader so C11 can measure what the page's anonymization costs

## C11: Train and evaluate on unseen phrases (owner: the project owner leads; ai-lead supports)
- [ ] Contrastive training where "same class" = same person across different sentences
- [ ] Report val AND test EER, mean +/- std over 3+ seeds, for: unseen people on unseen sentences; login-style
      (enroll 10 attempts of one phrase, verify more of it); cross-dataset (Aalto-trained model tested on CMU, no CMU training)
- [ ] With vs without key identity (the privacy cost of anonymized events)
- [ ] Compare with the current CMU-only encoder in docs/RESULTS.md; ship whichever wins on test EER

## C12: Multi-site service (owner: backend-developer)
- [ ] Sites table: site_id (= app_id), public site key, hashed secret, allowed origins; admin CLI to create sites
- [ ] Signed, single-use, 60 s tokens: the browser never receives "accepted", only a token
- [ ] POST /v1/siteverify {secret, token} -> {success, user_id, rhythm_match, risk}; customer server decides
- [ ] Enrollment tickets issued to the customer's server only after it authenticates the user (blocks enrolling a victim)
- [ ] CORS limited to each site's origins; opaque user ids (no emails)
- [ ] Failure-path tests: forged / expired / reused token, wrong secret, wrong origin, token for another user

## C13: Embeddable script + demo customer site (owner: frontend-developer, fullstack for wiring)
- [ ] keyauth.js: capture.js packaged as keyauth.attach(passwordField, {sitekey}), anonymized events, hidden token field
- [ ] Separate small demo "customer" site with its own users and hashed passwords that integrates keyauth end to end
- [ ] Browser e2e test of the integration (like C8's)
