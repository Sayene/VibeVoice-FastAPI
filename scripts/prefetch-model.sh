#!/usr/bin/env bash
#
# Pre-pull the VibeVoice model into the local HuggingFace cache so that
# subsequent `docker compose up` invocations start in seconds instead of
# downloading ~12 GB of weights every boot.
#
# Reads ./.env for VIBEVOICE_MODEL_PATH, HF_CACHE_DIR, and HF_TOKEN. Each
# may also be overridden inline:
#
#   ./scripts/prefetch-model.sh
#   VIBEVOICE_MODEL_PATH=aoi-ot/VibeVoice-Large ./scripts/prefetch-model.sh
#
# Resolution order for the downloader:
#   1. host `huggingface-cli` (fastest path; needs `pip install huggingface_hub`)
#   2. host `python3 -m huggingface_hub.commands.huggingface_cli` (no extra
#      install if huggingface_hub is already importable, e.g. inside a venv)
#   3. fallback: `docker run` against the prebuilt API image and use the
#      `huggingface_hub` python API there. Works even if the host has no
#      Python or huggingface_hub installed at all — just docker.
#
# Set HF_HUB_ENABLE_HF_TRANSFER=1 (default) to use the rust-based downloader
# (3-5× faster on Xet-backed repos). The script auto-disables it if the
# selected runtime cannot import hf_transfer (e.g. host python is PEP 668
# externally-managed and `pip install hf_transfer` is blocked).
#
# To keep the fast path on such hosts, force the docker fallback: it uses
# the API image where hf_transfer is preinstalled.
#   PREFETCH_VIA_DOCKER=1 ./scripts/prefetch-model.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Load .env if present, safely. We can't `source` it because values such as
# OPENAI_VOICE_MAPPING are JSON literals containing braces/commas/quotes that
# bash would try to execute as shell. Parse one key=value per line instead.
# Already-exported variables in the caller's environment win.
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        # Strip leading whitespace from key
        key="${key#"${key%%[![:space:]]*}"}"
        # Skip blanks and comments
        [[ -z "$key" ]] && continue
        [[ "$key" =~ ^# ]] && continue
        # Strip optional surrounding quotes from value
        if [[ "$value" =~ ^\"(.*)\"$ ]] || [[ "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi
        # Don't clobber an already-set variable from the caller
        if [[ -z "${!key+x}" ]]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
fi

MODEL_PATH="${VIBEVOICE_MODEL_PATH:-FabioSarracino/VibeVoice-Large-Q8}"
HF_CACHE="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HF_TOKEN_VAL="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
ENABLE_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Image used for the docker fallback. Override with PREFETCH_IMAGE=...
PREFETCH_IMAGE="${PREFETCH_IMAGE:-vibevoice-api:spark}"

mkdir -p "$HF_CACHE"

# Decide whether HF_HUB_ENABLE_HF_TRANSFER can actually be honoured by the
# python runtime that will perform the download. Without this probe the rust
# downloader bails out with a hard error if the package isn't importable.
probe_host_hf_transfer() {
    python3 -c "import hf_transfer" >/dev/null 2>&1
}

if [[ "$ENABLE_TRANSFER" == "1" ]]; then
    if command -v python3 >/dev/null 2>&1 && ! probe_host_hf_transfer; then
        # Only matters for the host CLI / host python paths. The docker
        # fallback uses the API image which ships hf_transfer (since v0.5.7).
        if command -v huggingface-cli >/dev/null 2>&1 \
           || python3 -c "import huggingface_hub" >/dev/null 2>&1; then
            echo "[!] hf_transfer not installed in host python — falling back to the"
            echo "    standard requests-based downloader (slower on Xet repos)."
            echo "    To enable the fast path:  pip install hf_transfer"
            ENABLE_TRANSFER=0
        fi
    fi
fi

echo "============================================================"
echo "VibeVoice model prefetch"
echo "============================================================"
echo "Model      : $MODEL_PATH"
echo "Cache dir  : $HF_CACHE"
echo "HF token   : $([[ -n "$HF_TOKEN_VAL" ]] && echo "set" || echo "(none — public model)")"
echo "hf_transfer: $ENABLE_TRANSFER"
echo "============================================================"

run_with_host_cli() {
    HF_HUB_ENABLE_HF_TRANSFER="$ENABLE_TRANSFER" \
    HF_TOKEN="$HF_TOKEN_VAL" \
        huggingface-cli download "$MODEL_PATH" --cache-dir "$HF_CACHE"
}

run_with_host_python() {
    HF_HUB_ENABLE_HF_TRANSFER="$ENABLE_TRANSFER" \
    HF_TOKEN="$HF_TOKEN_VAL" \
        python3 -m huggingface_hub.commands.huggingface_cli download \
            "$MODEL_PATH" --cache-dir "$HF_CACHE"
}

run_with_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: neither huggingface-cli nor docker found on PATH." >&2
        echo "Install one of them and re-run, or run inside an environment where" >&2
        echo "huggingface_hub is importable (e.g. an active venv)." >&2
        return 1
    fi
    if ! docker image inspect "$PREFETCH_IMAGE" >/dev/null 2>&1; then
        echo "ERROR: docker image '$PREFETCH_IMAGE' not found." >&2
        echo "Build it first:  docker compose -f docker-compose.spark.yml build" >&2
        echo "Or set PREFETCH_IMAGE to an image that has huggingface_hub installed." >&2
        return 1
    fi
    docker run --rm \
        -e HF_HUB_ENABLE_HF_TRANSFER="$ENABLE_TRANSFER" \
        -e HF_TOKEN="$HF_TOKEN_VAL" \
        -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL" \
        -v "$HF_CACHE:/root/.cache/huggingface:rw" \
        "$PREFETCH_IMAGE" \
        python3 -c "
import os, sys
from huggingface_hub import snapshot_download
try:
    snapshot_download('$MODEL_PATH', token=os.environ.get('HF_TOKEN') or None)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

if [[ "${PREFETCH_VIA_DOCKER:-0}" == "1" ]]; then
    echo "[1/1] PREFETCH_VIA_DOCKER=1 — using docker image '$PREFETCH_IMAGE'..."
    # The api image ships hf_transfer (requirements-api.txt), so the rust
    # fast path works there even when the host can't install it (PEP 668).
    run_with_docker
elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[1/1] Downloading via host huggingface-cli..."
    run_with_host_cli
elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "[1/1] huggingface-cli not on PATH — using host python3 + huggingface_hub..."
    run_with_host_python
else
    echo "[1/1] huggingface-cli not on PATH and host python3 lacks huggingface_hub —"
    echo "       falling back to docker image '$PREFETCH_IMAGE'..."
    run_with_docker
fi

echo
echo "Done. Cache populated at: $HF_CACHE"
echo "Subsequent 'docker compose up' will load from disk in <60s."
