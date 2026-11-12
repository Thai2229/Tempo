# tempo demo: the API and the web login page in one container.
#
#   docker build -t tempo .
#   docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" tempo
#
# The model is not baked into the image (checkpoints/ is gitignored and built
# from the CMU benchmark): train it first with
#   python scripts/get_cmu.py && python -m tempo.train --data cmu
# Without a checkpoint the server still starts and the page says "model not loaded".
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TEMPO_HOST=0.0.0.0 \
    TEMPO_PORT=8000 \
    TEMPO_CHECKPOINT=/app/checkpoints/model.pt \
    TEMPO_DB=/app/state/tempo.db

WORKDIR /app

# CPU-only PyTorch first, in its own layer, so code changes do not reinstall it.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url "$TORCH_INDEX_URL"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[app]" \
    && useradd --create-home --uid 10001 tempo \
    && mkdir -p /app/state /app/checkpoints \
    && chown tempo:tempo /app/state

USER tempo
VOLUME ["/app/state"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["tempo-demo"]
