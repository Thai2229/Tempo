"""`python -m keyauth.api` runs the server (KEYAUTH_HOST / KEYAUTH_PORT, default 127.0.0.1:8000)."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("keyauth.api:app", host=os.environ.get("KEYAUTH_HOST", "127.0.0.1"), port=int(os.environ.get("KEYAUTH_PORT", "8000")))
