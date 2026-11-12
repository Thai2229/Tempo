# typing-rhythm-login (tempo)

Log in by the rhythm of your typing: keystroke-dynamics biometrics with PyTorch.
Package `tempo` in src/tempo. Roadmap and status: docs/PLAN.md.

## Commands
- Install: `pip install -e ".[dev,app]"`
- Tests: `pytest` (must pass before any PR)
- Train without the dataset: `python -m tempo.train --data synthetic --epochs 5`
- Train on CMU: `python -m tempo.train --data cmu` (needs data/DSL-StrongPasswordData.csv)

## Data
- data/ and checkpoints/ are gitignored, so a cloud session starts without the CMU CSV or a trained model.
- Get the CMU benchmark with `python scripts/get_cmu.py` once it exists (milestone C6). If the download is blocked, use `--data synthetic` and say so in the PR; never report synthetic numbers as CMU numbers.
- Never commit biometric data, user templates, *.db or *.pt files.

## Code style
- Make minimal, targeted changes. Preserve existing variable names and module layout.
- Combine validation conditions into a single `if` statement.
- Do not use ternary operators.
- Keep existing `Lib.debug` statements.
- Type hints everywhere. Timings: milliseconds for raw events, seconds for features (see config.py).

## Workflow
- One milestone from docs/PLAN.md per session. Work on a branch, open a PR, tick the milestone's boxes in PLAN.md in the same PR.
- Model changes: report val AND test EER (unseen users) before and after, in a table. Pick thresholds on val only.
- If a requirement is ambiguous, write the question in the PR and make the most reasonable assumption.

## Delegation
- Delegate to a subagent whenever a task matches its description; don't do specialist work yourself when a matching agent exists.
- Model, features, training, evaluation: ai-lead (it may follow llm-architect / ai-engineer / prompt-engineer guidance where relevant).
- API (FastAPI), storage, auth flow: backend-developer. Web login page and keystroke capture: frontend-developer. End-to-end features spanning both: fullstack. Type-heavy TS work: typescript-pro.
- In every PR, list which subagents you used and for what.

## Safety
- Never commit secrets or keys. Use environment variables and .env.example.
- Security-sensitive code (verification, thresholds, lockout, replay checks) needs tests for the failure paths, not just the happy path.
