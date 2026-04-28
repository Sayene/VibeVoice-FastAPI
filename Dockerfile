FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_BUILD_ISOLATION=0 \
    PIP_ONLY_BINARY=:all: \
    MAX_JOBS=2 \
    MAKEFLAGS="-j2"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /app/venv

# Activate venv and upgrade pip
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY pyproject.toml README.md ./
COPY requirements-api.txt ./

# Install PyTorch with CUDA support (CUDA 12.8 compatible)
RUN pip install torch==2.8.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install torchao for INT8 quantization support
# torchao 0.13.0 was built for PyTorch 2.8.0
RUN pip install torchao==0.13.0

# Install bitsandbytes — required by pre-quantized HF checkpoints (e.g.
# FabioSarracino/VibeVoice-Large-Q8, DevParker/VibeVoice7b-low-vram) that
# ship a BitsAndBytes quantization_config in their config.json.
RUN pip install "bitsandbytes>=0.45" || \
    echo "WARNING: bitsandbytes install failed; bnb-quantized checkpoints will not load"

# Install flash-attn from pre-built wheel (PyTorch 2.8 compatible)
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl || \
    echo "WARNING: flash-attn wheel install failed, continuing without it"

# Install VibeVoice package (with --only-binary to prevent any compilation)
COPY vibevoice/ ./vibevoice/
COPY demo/ ./demo/
RUN pip install --only-binary=:all: -e . || pip install -e .

# Install API dependencies (with --only-binary safeguard)
RUN pip install --only-binary=:all: -r requirements-api.txt || pip install -r requirements-api.txt

# Copy API code
COPY api/ ./api/
COPY start.sh ./

# Create directories for voices and models
RUN mkdir -p /app/voices /app/models

# Expose API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the API (using venv python)
CMD ["sh", "-c", "/app/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8001}"]

