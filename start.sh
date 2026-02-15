#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        key=$(echo "$key" | xargs)
        [ -n "$key" ] && export "$key=$value"
    done < .env
    set +a
fi

# Default values
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8001}
WORKERS=${API_WORKERS:-1}
LOG_LEVEL=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')

echo "============================================================"
echo "Starting VibeVoice API Server"
echo "============================================================"
echo ""
echo "Server: http://$HOST:$PORT"
echo "API Docs: http://$HOST:$PORT/docs"
echo "Workers: $WORKERS"
echo "Log Level: $LOG_LEVEL"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

# Start the server
uvicorn api.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $LOG_LEVEL


