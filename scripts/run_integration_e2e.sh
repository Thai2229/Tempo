#!/usr/bin/env bash
# Starts keyauth (multi-site service) and the demo customer site on throw-away databases, then runs the browser e2e.
# Usage: scripts/run_integration_e2e.sh      (from anywhere; needs checkpoints/model.pt, playwright, chromium)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORK="$(mktemp -d)"
KA_PORT="${KEYAUTH_PORT:-8765}"
SITE_PORT="${DEMO_SITE_PORT:-9000}"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$WORK"; }
trap cleanup EXIT

export KEYAUTH_ENABLE_V1=1
KEYAUTH_TOKEN_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export KEYAUTH_TOKEN_KEY
export KEYAUTH_DB="$WORK/keyauth.db" KEYAUTH_PORT="$KA_PORT"

CREATED="$(python -m keyauth.sites create --name "Demo Shop" --origin "http://127.0.0.1:$SITE_PORT")"
SITE_KEY="$(printf '%s\n' "$CREATED" | sed -n 's/^site key: \(pk_[^ ]*\).*/\1/p')"
SITE_SECRET="$(printf '%s\n' "$CREATED" | sed -n 's/^secret: *\(sk\.[^ ]*\).*/\1/p')"

python -m keyauth.api >"$WORK/keyauth.log" 2>&1 &
PIDS+=($!)
KEYAUTH_URL="http://127.0.0.1:$KA_PORT" KEYAUTH_SITE_KEY="$SITE_KEY" KEYAUTH_SITE_SECRET="$SITE_SECRET" \
  DEMO_SITE_DB="$WORK/site.db" DEMO_SITE_PORT="$SITE_PORT" python -m demo_site >"$WORK/site.log" 2>&1 &
PIDS+=($!)

for url in "http://127.0.0.1:$KA_PORT/v1/keyauth.js" "http://127.0.0.1:$SITE_PORT/login"; do
  for _ in $(seq 1 60); do curl -fs -o /dev/null --noproxy '*' "$url" && break; sleep 0.5; done
done

set +e
SITE_URL="http://127.0.0.1:$SITE_PORT" KEYAUTH_URL="http://127.0.0.1:$KA_PORT" KEYAUTH_SITE_SECRET="$SITE_SECRET" \
  node tests/e2e/integration.e2e.mjs
STATUS=$?
echo "---- demo site log (flags and second-step codes) ----"
grep -E "FLAG|SECOND-STEP|decision=" "$WORK/site.log" | sed -E 's/code=[0-9]{6}/code=******/' || true
exit $STATUS
