# typing-rhythm-login

Log in by the rhythm of your typing. `keyauth` is a keystroke-dynamics biometric: it learns how *you* type your
password (how long you hold each key, how quickly you move to the next) and checks that rhythm at login, as a second
factor next to the password itself.

![Demo: enrolling with 10 typed samples, then logging in and being accepted](docs/demo.gif)

*The demo page. The GIF was recorded with a scripted browser, not a person: it enrolls with 10 attempts, then
logs in. Debug mode shows the score against the threshold.*

## How it works

1. **Capture.** The browser records `keydown` / `keyup` times with `performance.now()` (milliseconds). Before
   anything is sent, the page replaces every key with an anonymous placeholder (`x`, `k0`, `k1`, ...), so **the
   server never learns what was typed**. Only the timings are sent. Modifier keys and Enter keep their names.
2. **Features.** Events are paired into keystrokes, and each keystroke becomes three timings in seconds: `hold`
   (down to up), `dd` (down to next down) and `ud` (up to next down). This is the layout of the CMU Keystroke
   Dynamics Benchmark (`src/keyauth/events.py`, `features.py`).
3. **Embedding.** A small PyTorch encoder (Conv1d, then a bidirectional GRU, then mean and max pooling, then a
   linear layer) maps one attempt to a point on the unit sphere. It is trained with a supervised contrastive loss so
   that attempts by the same person land close together (`src/keyauth/model/`). New users need no retraining.
4. **Enrollment.** The user types the phrase 10 times. The template is the normalized mean of those embeddings.
   Enrollment restarts if the user's own attempts disagree too much.
5. **Verification.** Accept if the distance to the template is at most a threshold chosen on validation users.
   The engine also does the following (`src/keyauth/auth/engine.py`):
   - rejects replays, since a human never types identical timings twice;
   - locks the account after repeated failures;
   - nudges the template toward confident accepts, because rhythm drifts over time.

   Optional extras:
   - per-user thresholds (C6b);
   - the classic scaled-Manhattan and Mahalanobis detectors instead of the encoder.

The engine never sees the password. The app checks the password itself, and keyauth answers only "does the
rhythm match?".

## Results

Real data: CMU Keystroke Dynamics Benchmark (51 typists, password `.tie5Roanl`).

| Setting | Value |
|---|---|
| Enrollment | 10 attempts |
| Test users | unseen during training |
| Threshold | one global threshold, picked on val users only |
| Metric | equal error rate (EER, lower is better), mean ± std over 5 user splits |

| method | val EER | test EER |
|---|---|---|
| Scaled Manhattan (classic baseline) | 23.72 ± 1.57 % | 22.14 ± 5.98 % |
| Mahalanobis (classic baseline) | 22.45 ± 2.62 % | 23.27 ± 4.89 % |
| Encoder before C6 | 19.93 ± 5.81 % | 28.28 ± 7.14 % |
| **Encoder, shipped** | 18.76 ± 4.47 % | **21.50 ± 9.30 %** |

Honest reading:
- **It's a tie, not a win.** The shipped encoder ties the classic detectors, which have been around since 2009.
- **Splits vary a lot.** With only 8 test users per split, treat any single number as ± 5 points.
- **Per-user thresholds trade one error for another.** On split seed 0 they cut the false-reject rate from 36.8 %
  to 31.8 % but raise false accepts from 13.9 % to 15.0 %, so they are off by default.
- **Short enrollment costs accuracy.** 10 enrollment attempts is far fewer than the 200 that Killourhy & Maxion
  (2009) used.

On its own this is not strong enough to authenticate anyone. Use it as a soft signal next to a password.

Details, per-seed tables, the val/test gap investigation and known failure cases: [docs/RESULTS.md](docs/RESULTS.md).

## Run it

Needs Python 3.10+.

```bash
pip install -e ".[dev,app]"
```

```bash
python scripts/get_cmu.py
```

```bash
python -m keyauth.train --data cmu
```

```bash
python -m keyauth.api
```

`get_cmu.py` downloads the CMU benchmark into `data/`. Training takes about 2 minutes on a laptop CPU and writes
`checkpoints/model.pt`. The last command serves the API and the demo page.

Then open http://127.0.0.1:8000/. To see the score and threshold in the page's debug mode, start the server with
`KEYAUTH_DEBUG_SCORES=1`. All settings are environment variables, listed in [.env.example](.env.example).

No CMU download? You can train on synthetic typists instead, to try the pipeline. The numbers it reports are not
CMU numbers.

```bash
python -m keyauth.train --data synthetic --epochs 5
```

### Docker

The model is not baked into the image. Train it first (above), then mount `checkpoints/`:

```bash
docker build -t keyauth .
```

```bash
docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" -v keyauth-state:/app/state keyauth
```

The container runs as a non-root user and keeps its SQLite database in the `keyauth-state` volume.

### API

| Endpoint | Purpose |
|---|---|
| `POST /enroll` `{user_id, events \| timings}` | add one enrollment attempt; returns progress |
| `POST /verify` `{user_id, events \| timings}` | `{accepted, reason}`; locked accounts get 423 with `Retry-After` |
| `GET /status?user_id=` | enrollment and lock status |
| `DELETE /user?user_id=` | delete a user's template (set `KEYAUTH_ADMIN_TOKEN` outside the demo) |
| `GET /health` | liveness and whether the model loaded |

Every error has the shape `{"error": {"code", "message"}}`. The full status-code mapping is in
`src/keyauth/api/__init__.py`. Requests are rate-limited per client IP and per user. Logs carry a hash of
`user_id` and the outcome, never timings or scores.

## Tests

```bash
python -m pytest
```

This includes the JS tests of the capture code, which are skipped without Node. CI runs it on every push and pull
request (`.github/workflows/ci.yml`). The browser end-to-end check runs against a live server; its usage is in
`tests/e2e/demo.e2e.mjs`.

## Project layout

```
src/keyauth/
  events.py, features.py   raw key events -> (hold, dd, ud) timings
  data/                    CMU and synthetic loaders, user-disjoint splits
  model/                   RhythmEncoder and losses
  train.py, evaluate.py    training, EER, checkpoints
  baselines.py             scaled Manhattan and Mahalanobis
  thresholds.py            per-user thresholds
  auth/                    engine, SQLite template store, validation, scorers
  api/                     FastAPI app
  web/static/              demo page (plain HTML and JS, no build step)
scripts/                   get_cmu.py, compare_models.py
docs/                      PLAN.md (roadmap), RESULTS.md (all numbers)
```

## Privacy

- Keystroke timings are biometric data. The repo never contains user templates, databases or model files
  (`.gitignore`).
- The demo page sends anonymized events only.
- `DELETE /user` removes a user's template and samples.

## License

MIT. The CMU benchmark data is © Killourhy & Maxion, Carnegie Mellon University. It is downloaded from their site
and not redistributed here.
