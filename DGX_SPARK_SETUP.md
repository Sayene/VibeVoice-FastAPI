# VibeVoice on NVIDIA DGX Spark — Setup Guide

Step-by-step guide to run the VibeVoice FastAPI server on a DGX Spark (ARM64
Grace-Blackwell) remote machine.

The Spark differs from a standard x86_64 Linux GPU host in two ways that matter
here:

1. **CPU is aarch64 (ARM64)** — most Python wheels hosted on PyPI for CUDA are
   x86_64 only (including the `flash-attn` wheel we use on standard hosts).
2. **GPU is Blackwell (SM 10.0 / 12.0)** — needs PyTorch built against CUDA 12.8+.

The solution below sidesteps both by using the NVIDIA NGC PyTorch container
(`nvcr.io/nvidia/pytorch:25.03-py3`), which ships multi-arch images with
PyTorch, CUDA 12.8, cuDNN, and flash-attention prebuilt for aarch64 + Blackwell.

---

## 1. Prerequisites on the Spark

SSH into the Spark and verify:

```bash
ssh user@spark-host

# Confirm we're on ARM64
uname -m          # expect: aarch64

# Confirm the GPU is visible. On a Spark you should see:
#   NVIDIA GB10 ... Driver Version: 580.x ... CUDA Version: 13.0
# "Memory-Usage: Not Supported" is normal — GB10 uses unified LPDDR5X, not
# discrete VRAM, so nvidia-smi cannot report it.
nvidia-smi

# Confirm Docker + NVIDIA container toolkit are installed
docker version
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If the last command fails, install the NVIDIA Container Toolkit:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

You will also need an NGC login to pull `nvcr.io/nvidia/pytorch`. Create a
free account at https://ngc.nvidia.com, generate an API key, then:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API key>
```

## 2. Clone the repo

```bash
git clone https://github.com/<your-fork>/VibeVoice-FastAPI.git
cd VibeVoice-FastAPI
```

## 3. Create the `.env` file

```bash
cp docker-env.example .env
```

Edit `.env` and set at minimum:

```bash
# HuggingFace model ID — will auto-download to HF_CACHE_DIR on first run
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B

# Blackwell supports bfloat16 natively
VIBEVOICE_DTYPE=bfloat16

# NGC image has flash-attn built for aarch64+Blackwell
VIBEVOICE_ATTN_IMPLEMENTATION=flash_attention_2

# Host paths for persistent caches — use absolute paths, not ~
HF_CACHE_DIR=/home/YOUR_USER/.cache/huggingface
MODELS_DIR=/home/YOUR_USER/vibevoice-models
VOICES_DIR=/home/YOUR_USER/vibevoice-voices

API_PORT=8001
```

Create the host directories:

```bash
mkdir -p ~/.cache/huggingface ~/vibevoice-models ~/vibevoice-voices
# Seed with the bundled demo voices so you have something to start with
cp demo/voices/* ~/vibevoice-voices/
```

## 4. Build and launch

```bash
# First build pulls ~15 GB NGC image; subsequent builds are fast.
docker compose -f docker-compose.spark.yml build

docker compose -f docker-compose.spark.yml up -d
```

Watch the logs — first launch downloads the model (~3 GB for VibeVoice-1.5B):

```bash
docker compose -f docker-compose.spark.yml logs -f
```

Wait for `API server ready!`.

## 5. Verify

From the Spark (or from your laptop if port 8001 is reachable):

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/vibevoice/voices
curl http://localhost:8001/v1/vibevoice/health
```

Interactive API explorers (auto-generated from the FastAPI schema):

| URL                                   | What it is                                    |
|---------------------------------------|-----------------------------------------------|
| `http://<spark-host>:8001/docs`       | Swagger UI — try requests straight from the browser, including multipart file upload to `POST /v1/vibevoice/voices`. |
| `http://<spark-host>:8001/redoc`      | ReDoc — cleaner reference-style documentation. |
| `http://<spark-host>:8001/openapi.json` | Raw OpenAPI 3.1 schema for client codegen.   |

If port 8001 isn't reachable from your laptop, open an SSH tunnel:
`ssh -L 8001:localhost:8001 user@spark-host` then visit
`http://localhost:8001/docs`.

Generate test audio:

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello from a DGX Spark.","voice":"alloy"}' \
  --output hello.mp3
```

## 6. Manage custom voices over HTTP

Once the server is running, voice presets can be managed via the API (no SSH
or container restart required). See the section below for full endpoint docs.

```bash
# List
curl http://localhost:8001/v1/vibevoice/voices

# Upload (name defaults to filename stem if omitted)
curl -X POST http://localhost:8001/v1/vibevoice/voices \
  -F "file=@/path/to/my_voice.wav" \
  -F "name=my_custom_voice"

# Delete
curl -X DELETE http://localhost:8001/v1/vibevoice/voices/my_custom_voice

# Rescan voices dir from disk (if you dropped files in via scp/rsync)
curl -X POST http://localhost:8001/v1/vibevoice/voices/reload
```

## 7. Updating / restarting

```bash
git pull
docker compose -f docker-compose.spark.yml up -d --build
```

---

## Troubleshooting

- **`exec format error` on container start** — the image was built for the wrong
  arch. Rebuild with `--no-cache` on the Spark itself (don't `docker save` from
  an x86_64 host and load onto the Spark).
- **`flash_attn not available`** — fall back by setting
  `VIBEVOICE_ATTN_IMPLEMENTATION=sdpa` in `.env`. Slower but always works.
- **`CUDA error: no kernel image is available for execution on the device`** —
  base image's PyTorch is too old for GB10/Blackwell (needs SM 10.0). Bump the
  NGC tag in `Dockerfile.spark` to a newer `nvcr.io/nvidia/pytorch:YY.MM-py3`
  (25.10 or later recommended for GB10).
- **OOM during generation** — set `VIBEVOICE_QUANTIZATION=int8_torchao` in
  `.env`, or reduce `VIBEVOICE_INFERENCE_STEPS`.
- **Can't pull `nvcr.io/...`** — you skipped `docker login nvcr.io` (step 1).

---

## Voice management endpoint reference

| Method | Path                                   | Purpose                          |
|--------|----------------------------------------|----------------------------------|
| GET    | `/v1/vibevoice/voices`                 | List all registered presets      |
| POST   | `/v1/vibevoice/voices`                 | Upload a new preset (multipart)  |
| DELETE | `/v1/vibevoice/voices/{name}`          | Remove a preset                  |
| POST   | `/v1/vibevoice/voices/reload`          | Rescan the voices directory      |

Upload form fields:

- `file` (required) — audio file. Accepted: `.wav .mp3 .flac .ogg .m4a .aac`.
- `name` (optional) — preset name. Defaults to the uploaded file's stem.

The voices directory is mounted read-write from the host (`VOICES_DIR` in
`.env`), so uploads and deletes survive container restarts.
