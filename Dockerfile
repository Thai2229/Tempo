# keyauth demo: the API and the web login page in one container.
#
#   docker build -t keyauth .
#   docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" keyauth
#
# The model is not baked into the image (checkpoints/ is gitignored and built
# from the CMU benchmark): train it first with
#   python scripts/get_cmu.py && python -m keyauth.train --data cmu
# Without a checkpoint the server still starts and the page says "model not loaded".
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KEYAUTH_HOST=0.0.0.0 \
    KEYAUTH_PORT=8000 \
    KEYAUTH_CHECKPOINT=/app/checkpoints/model.pt \
    KEYAUTH_DB=/app/state/keyauth.db

WORKDIR /app

# CPU-only PyTorch first, in its own layer, so code changes do not reinstall it.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url "$TORCH_INDEX_URL"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[app]" \
    && useradd --create-home --uid 10001 keyauth \
    && mkdir -p /app/state /app/checkpoints \
    && chown keyauth:keyauth /app/state

USER keyauth
VOLUME ["/app/state"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["keyauth-demo"]
