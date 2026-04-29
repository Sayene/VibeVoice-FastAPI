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
# By default the rust-based hf_transfer downloader (3-5× faster on Xet
# repos) is used ONLY when the script can verify the runtime importing
# `hf_transfer` succeeds. If the package isn't available, the standard
# requests-based downloader is used (slower but never errors). Set
# HF_HUB_ENABLE_HF_TRANSFER=1 explicitly to force-enable; the script will
# still skip it if it cannot find the package, to avoid the
# "Fast download ... but 'hf_transfer' package is not available" hard
# error that huggingface_hub raises when the env var is set without the
# package installed.
#
# Tip on PEP 668 externally-managed hosts: set PREFETCH_VIA_DOCKER=1 to
# run the download inside the API image, which already ships hf_transfer:
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

# Image used for the docker fallback. Override with PREFETCH_IMAGE=...
PREFETCH_IMAGE="${PREFETCH_IMAGE:-vibevoice-api:spark}"

mkdir -p "$HF_CACHE"

# `huggingface_hub` raises a hard error if HF_HUB_ENABLE_HF_TRANSFER=1 but
# the `hf_transfer` package isn't importable in the runtime that performs
# the download (and that runtime can be a different python than `python3`
# on PATH — e.g. pipx-installed huggingface-cli has its own venv). To
# avoid that: explicitly set 0 unless we can verify the package is
# importable in the runtime we'll actually use.
#
# Note we DO NOT inherit the user's exported HF_HUB_ENABLE_HF_TRANSFER; we
# decide per-runtime below and pass it inline to the child process.
ENABLE_TRANSFER="0"
unset HF_HUB_ENABLE_HF_TRANSFER

probe_runtime_has_hf_transfer() {
    # Run the same python the cli would run. We approximate by trying:
    #   1. `huggingface-cli env` (some HF versions expose hf_transfer status)
    #   2. The python interpreter the cli's shebang points at, if we can
    #      find it.
    #   3. Plain `python3 -c "import hf_transfer"` as last resort.
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c "import hf_transfer" >/dev/null 2>&1; then
            return 0
        fi
    fi
    if command -v huggingface-cli >/dev/null 2>&1; then
        local cli_py
        cli_py="$(head -n1 "$(command -v huggingface-cli)" 2>/dev/null \
                  | sed -n 's|^#! *\(/[^ ]*\).*|\1|p')"
        if [[ -n "$cli_py" && -x "$cli_py" ]]; then
            if "$cli_py" -c "import hf_transfer" >/dev/null 2>&1; then
                return 0
            fi
        fi
    fi
    return 1
}

run_with_host_cli() {
    if probe_runtime_has_hf_transfer; then ENABLE_TRANSFER=1; fi
    print_banner
    HF_HUB_ENABLE_HF_TRANSFER="$ENABLE_TRANSFER" \
    HF_TOKEN="$HF_TOKEN_VAL" \
        huggingface-cli download "$MODEL_PATH" --cache-dir "$HF_CACHE"
}

run_with_host_python() {
    if probe_runtime_has_hf_transfer; then ENABLE_TRANSFER=1; fi
    print_banner
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
    # The api image ships hf_transfer (requirements-api.txt since v0.5.7),
    # so we can safely turn it on.
    if docker run --rm "$PREFETCH_IMAGE" python3 -c "import hf_transfer" \
            >/dev/null 2>&1; then
        ENABLE_TRANSFER=1
    fi
    print_banner
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

print_banner() {
    echo "============================================================"
    echo "VibeVoice model prefetch"
    echo "============================================================"
    echo "Model      : $MODEL_PATH"
    echo "Cache dir  : $HF_CACHE"
    echo "HF token   : $([[ -n "$HF_TOKEN_VAL" ]] && echo "set" || echo "(none — public model)")"
    if [[ "$ENABLE_TRANSFER" == "1" ]]; then
        echo "hf_transfer: enabled (rust fast path)"
    else
        echo "hf_transfer: disabled (standard downloader; install hf_transfer to speed up Xet repos)"
    fi
    echo "============================================================"
}

if [[ "${PREFETCH_VIA_DOCKER:-0}" == "1" ]]; then
    echo "[1/1] PREFETCH_VIA_DOCKER=1 — using docker image '$PREFETCH_IMAGE'..."
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
