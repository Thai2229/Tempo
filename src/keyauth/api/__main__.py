"""`python -m keyauth.api` (or `keyauth-demo`) runs the server and the demo page (KEYAUTH_HOST / KEYAUTH_PORT, default 127.0.0.1:8000)."""

import os
from pathlib import Path

import uvicorn


def main() -> None:
    host = os.environ.get("KEYAUTH_HOST", "127.0.0.1")
    port = int(os.environ.get("KEYAUTH_PORT", "8000"))
    checkpoint = os.environ.get("KEYAUTH_CHECKPOINT", "checkpoints/model.pt")
    if os.environ.get("KEYAUTH_SERVE_UI", "1").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"keyauth demo: http://{host}:{port}/", flush=True)
    if not Path(checkpoint).is_file():
        print(f"warning: no model at {checkpoint}; the page will show 'model not loaded'.", flush=True)
        print("  fix: python scripts/get_cmu.py && python -m keyauth.train --data cmu", flush=True)
    uvicorn.run("keyauth.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
