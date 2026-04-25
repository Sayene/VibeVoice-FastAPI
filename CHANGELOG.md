# Changelog

All notable changes to this fork of VibeVoice-FastAPI are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-04-25

### Added

- **Sentence-aware chunking for long generations** — long scripts are now split
  at sentence boundaries and synthesized sequentially, avoiding the quality
  degradation VibeVoice exhibits on very long single-shot generations. New
  helper `api/services/chunking.py` ports the strategy from VibeVoice-ComfyUI:
  - Splits on sentence terminators (`.!?`), falls back to period-based
    splitting, and sub-splits oversized sentences at `,`/`;`.
  - Multi-speaker scripts are chunked **per-turn**: each chunk re-emits its
    `Speaker N:` label and never crosses speaker boundaries.
  - Chunks are concatenated in non-streaming mode (with optional silence pad);
    in streaming mode each chunk's audio is forwarded sequentially over the
    same SSE response.
- **New generation parameters on `POST /v1/vibevoice/generate`**:
  - `do_sample` (bool) — switch between greedy and sampling decoding.
    Auto-enabled when `temperature` or `top_p` is provided.
  - `temperature` (0.0–5.0) — sampling temperature.
  - `top_p` (0.0–1.0) — nucleus sampling.
  - `max_words_per_chunk` (0–2000) — per-request override; `0` disables
    chunking. Defaults to server `default_max_words_per_chunk`.
  - `chunk_silence_ms` (0–5000) — silence inserted between concatenated chunks
    in non-streaming mode.
- **New settings / env vars**:
  - `DEFAULT_MAX_WORDS_PER_CHUNK` (default `250`) — server-wide chunk size.
  - `DEFAULT_CHUNK_SILENCE_MS` (default `0`) — server-wide chunk gap.
- The OpenAI-compatible `POST /v1/audio/speech` endpoint also benefits: it
  uses the server defaults for chunking and sampling, so long inputs work
  without client changes.

### Changed

- `TTSService.generate_speech()` signature extended with `do_sample`,
  `temperature`, `top_p`, `max_words_per_chunk`, `chunk_silence_ms`. Existing
  callers continue to work (all new params are optional). Internals refactored
  into `_build_inputs`, `_build_generation_config`, `_generate_full`, and
  `_generate_streaming_chunks` to share code between single-shot and chunked
  paths.
- Generation logs now include sampling mode, temperature, top_p, and the
  active `max_words_per_chunk`.

## [0.3.5] — 2026-04-25

### Added

- **`language` parameter on `POST /v1/audio/speech`** — the OpenAI-compatible
  endpoint now accepts an optional ISO 639-1 code (e.g. `"pl"`, `"de"`). When
  supplied, the voice is resolved to a preset stored under
  `<voices_dir>/<language>/`, so the same OpenAI voice name (`alloy`, `echo`, …)
  or shared preset stem can produce output in the requested language's accent.
  - New `VoiceManager.resolve_voice_for_language()` handles the lookup,
    including stripping legacy `<code>-` prefixes from the default OpenAI voice
    mapping (e.g. `en-Alice_woman` → matches `pl/Alice_woman`).
  - Requests with a `language` that has no matching preset return HTTP 400 and
    list the presets available in that language.
  - Without `language`, behavior is unchanged (OpenAI mapping, then direct
    preset name).

## [0.3.4] — 2026-04-24

### Fixed

- **Reference voice audio bleeding into generated output** — long or
  silence-padded reference clips caused the model to reproduce fragments of
  the reference file in the synthesised speech. Fixed by pre-processing every
  reference clip through `prepare_voice_sample()` before it reaches the model:
  1. Leading/trailing silence stripped via `librosa.effects.trim` (default
     threshold: 30 dB).
  2. Clip capped at `VOICE_SAMPLE_MAX_DURATION` seconds (default: 10 s).
  Applied to all entry points — on-disk presets loaded by `VoiceManager` and
  base64 inline clips sent in the `/v1/vibevoice/generate` request body.

### Added

- Three new env/settings knobs (documented in `docker-env.example`):
  - `VOICE_SAMPLE_MAX_DURATION` (default `10.0`) — max seconds of reference audio.
  - `VOICE_SAMPLE_TRIM_SILENCE` (default `true`) — enable silence trimming.
  - `VOICE_SAMPLE_TRIM_DB` (default `30.0`) — dB threshold for silence trim.

## [0.3.3] — 2026-04-24

### Added

- **Optional `bitsandbytes` install on Spark** — `Dockerfile.spark` now attempts
  `pip install "bitsandbytes>=0.45"` (alongside torchao) so pre-quantized
  bnb-8bit HF checkpoints such as `FabioSarracino/VibeVoice-Large-Q8` load
  out of the box. Install is best-effort; if no aarch64 wheel matches, the
  build continues and bnb-quantized models will fail with a clear error
  while full-precision / torchao paths still work.

### Notes

- On the Spark (128 GB unified memory) there is usually no need for bnb int8.
  Using `rsxdalv/VibeVoice-Large` or `microsoft/VibeVoice-1.5B` in bf16 is
  faster and avoids the bnb-on-aarch64 dependency entirely.

## [0.3.2] — 2026-04-24

### Fixed

- **`ModuleNotFoundError: No module named 'transformers'` on Spark startup** —
  `Dockerfile.spark` was installing the `vibevoice` package with `--no-deps`
  (an over-cautious attempt to protect the NGC-provided torch), which also
  skipped `transformers`, `accelerate`, `diffusers`, `librosa`, and `numba`.
  `torch` in `pyproject.toml` has no version pin, so `pip install -e .`
  without `--no-deps` accepts the NGC torch and installs the rest.

## [0.3.1] — 2026-04-24

### Added

- **HuggingFace auth forwarding on Spark**
  - `docker-compose.spark.yml` now forwards `HF_TOKEN` (and the legacy
    `HUGGING_FACE_HUB_TOKEN` alias) from `.env` into the container, so gated
    models such as `microsoft/VibeVoice-1.5B` download on first launch
    without an interactive `huggingface-cli login`.
  - Setup guide documents how to generate a token, accept the gated-repo
    license, and verify the token against `whoami-v2` before building.
- Swagger UI / ReDoc / OpenAPI endpoints documented in the DGX Spark guide,
  with an SSH tunnel recipe for remote access.

## [0.3.0] — 2026-04-24

### Added

- **Recursive voices directory with language subfolders**
  - `VoiceManager` now walks the voices directory recursively (`rglob`), so
    presets can be organized by language (e.g. `voices/en/woman_1_en.mp3`,
    `voices/pl/man_2_pl.wav`).
  - Language detection precedence: parent folder code (`en/`, `pl/`) →
    trailing `_<code>` stem suffix (`woman_1_en`) → legacy `<code>-` prefix
    (`en-Alice_woman`). Extensible `_LANG_CODES` map covers common ISO 639-1
    codes.
  - Stem collisions across subfolders fall back to folder-qualified names
    (e.g. `pl/woman_1`) so nothing is silently overwritten.
  - `POST /v1/vibevoice/voices` accepts an optional `language` form field;
    when provided, the upload is stored under `<voices_dir>/<language>/`.

## [0.2.0] — 2026-04-24

First release of the [Sayene/VibeVoice-FastAPI](https://github.com/Sayene/VibeVoice-FastAPI)
fork. Adds NVIDIA DGX Spark (ARM64 / GB10 Grace-Blackwell) deployment support
and HTTP endpoints for managing custom voice presets at runtime.

### Added

- **DGX Spark / ARM64 support**
  - `Dockerfile.spark` — new build targeting aarch64 + Blackwell. Uses the NGC
    PyTorch container (`nvcr.io/nvidia/pytorch:25.10-py3`) so PyTorch, CUDA 13,
    cuDNN and flash-attention come prebuilt for GB10 (SM 10.0/12.0). This
    avoids the x86_64-only `flash-attn` wheel and ARM wheel-availability
    issues of the original `nvidia/cuda` base.
  - `docker-compose.spark.yml` — Spark-specific compose file: pins
    `platform: linux/arm64`, reserves all GPUs, mounts the voices volume
    read-write, sets a 16 GB shm, defaults to `bfloat16` +
    `flash_attention_2`.
  - `DGX_SPARK_SETUP.md` — step-by-step remote setup guide (prereqs, NGC
    login, `.env`, build, verify, voice CRUD examples, troubleshooting).

- **Voice management endpoints** on `/v1/vibevoice`:
  - `POST /v1/vibevoice/voices` — multipart upload (`file`, optional `name`).
    Validates the upload by decoding it, rolls back on failure, returns 409
    on duplicate names.
  - `DELETE /v1/vibevoice/voices/{voice_name}` — removes a preset from disk
    and unregisters it. Guards against path escape outside the voices dir.
  - `POST /v1/vibevoice/voices/reload` — rescans the voices directory so
    files added externally (scp/rsync) become available without a restart.
  - Supporting helpers `add_voice_from_bytes`, `delete_voice`, `reload` on
    `VoiceManager`.

### Changed

- `docker-compose.yml` — voices volume flipped from `:ro` to `:rw` so the new
  upload/delete endpoints can persist changes back to the host.
- `api/main.py` — root `/` endpoint now enumerates the new voice CRUD
  endpoints. API version bumped to `0.2.0`.

### Notes

- The `vibevoice` Python package (`pyproject.toml`) version is intentionally
  left at the upstream value — it tracks the Microsoft model package, not
  this API server.
- For GB10 Blackwell hosts, driver 580+ is required (verify with
  `nvidia-smi`). Older Spark images with driver <580 should pin the base
  image to `nvcr.io/nvidia/pytorch:25.03-py3` (CUDA 12.8) instead.

## [0.1.0] — upstream baseline

Initial state inherited from [ncoder-ai/VibeVoice-FastAPI](https://github.com/ncoder-ai/VibeVoice-FastAPI).
OpenAI-compatible TTS API over VibeVoice with x86_64 CUDA Docker support,
multi-speaker generation, streaming, and voice presets discovered from disk.
