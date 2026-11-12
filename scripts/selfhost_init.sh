#!/bin/sh
# Prepares a self-hosted Tempo: creates .env with generated secrets and the directories compose mounts.
# Never overwrites a value that is already in .env, never prints a secret.
#   sh scripts/selfhost_init.sh
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE="$ROOT/.env"

gen() {
    # 64 hex characters (256 bits) from the kernel RNG
    od -An -N48 -tx1 /dev/urandom | tr -d ' \n' | head -c 64
}

mkdir -p "$ROOT/checkpoints" "$ROOT/data"
umask 077
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

has_value() {
    grep -Eq "^$1=.+" "$ENV_FILE"
}

add() {
    # add NAME VALUE LABEL: append only if NAME has no value yet
    if has_value "$1"; then
        echo "kept    $1 (already in .env)"
    else
        grep -v "^$1=" "$ENV_FILE" > "$ENV_FILE.tmp" || true
        printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE.tmp"
        mv "$ENV_FILE.tmp" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        echo "created $1"
    fi
}

add TEMPO_TOKEN_KEY "$(gen)"
add TEMPO_ADMIN_TOKEN "$(gen)"
add TEMPO_UID "$(id -u)"
add TEMPO_GID "$(id -g)"

echo
echo ".env is ready (mode 600, gitignored). Next: docs/SELF_HOSTING.md"
