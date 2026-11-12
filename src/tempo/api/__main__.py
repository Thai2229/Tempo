"""`python -m tempo.api` (or `tempo-demo`) runs the server and the demo page (TEMPO_HOST / TEMPO_PORT, default 127.0.0.1:8000)."""

import os
from pathlib import Path

import uvicorn


def main() -> None:
    host = os.environ.get("TEMPO_HOST", "127.0.0.1")
    port = int(os.environ.get("TEMPO_PORT", "8000"))
    checkpoint = os.environ.get("TEMPO_CHECKPOINT", "checkpoints/model.pt")
    if os.environ.get("TEMPO_SERVE_UI", "1").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"tempo demo: http://{host}:{port}/", flush=True)
    if not Path(checkpoint).is_file():
        print(f"warning: no model at {checkpoint}; the page will show 'model not loaded'.", flush=True)
        print("  fix: python scripts/get_cmu.py && python -m tempo.train --data cmu", flush=True)
    uvicorn.run("tempo.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
