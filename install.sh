#!/bin/bash
# VibeVoice-FastAPI One-Click Installer
# Download and run: curl -fsSL https://raw.githubusercontent.com/ncoder-ai/VibeVoice-FastAPI/main/install.sh | bash
set -e

echo ""
echo "============================================================"
echo "  VibeVoice-FastAPI One-Click Installer"
echo "============================================================"
echo ""

# Check for git
if ! command -v git &> /dev/null; then
    echo "ERROR: git is not installed."
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install git"
    echo "  macOS: xcode-select --install"
    echo ""
    exit 1
fi

# Check for Python 3
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        PY_MAJOR=$($cmd -c 'import sys; print(sys.version_info.major)' 2>/dev/null)
        if [ "$PY_MAJOR" = "3" ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3 is not installed."
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install python3"
    echo "  macOS: brew install python"
    echo ""
    exit 1
fi

echo "Using: git ($(git --version))"
echo "Using: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
echo ""

# Clone or update repo
INSTALL_DIR="VibeVoice-FastAPI"
if [ -d "$INSTALL_DIR" ]; then
    echo "Found existing $INSTALL_DIR directory."
    echo "Updating..."
    git -C "$INSTALL_DIR" pull
else
    echo "Cloning VibeVoice-FastAPI..."
    git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git "$INSTALL_DIR"
fi

echo ""

# Run interactive installer with stdin from terminal
# (needed when this script is piped via curl | bash)
cd "$INSTALL_DIR"
$PYTHON_CMD install.py </dev/tty
