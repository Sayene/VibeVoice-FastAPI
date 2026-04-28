# Changelog

All notable changes to this fork of VibeVoice-FastAPI are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.5.4] — 2026-04-28

### Changed (default)

- **`DEFAULT_DO_SAMPLE` now defaults to `False`** (was `True`). Greedy
  decoding matches ComfyUI's `use_sampling=false` (the actual node-config
  in production use) and Microsoft's reference inference demo. Stochastic
  sampling lets the LLM drift away from the reference voice's pronunciation
  and was a major contributor to poor accent fidelity on non-English
  voices. `temperature` and `top_p` are unused while `do_sample=False`.
  The `OpenAITTSRequest` and `VibeVoiceGenerateRequest` Pydantic schemas
  now reflect this default in Swagger.

### Fixed

- **Pre-quantized VibeVoice checkpoints now load correctly** (e.g.
  `FabioSarracino/VibeVoice-Large-Q8`, `DevParker/VibeVoice7b-low-vram`).
  `TTSService.load_model` detects models that ship their own
  `quantization_config` (either embedded in `config.json` or in a
  `quantization_config.json` sidecar — same logic as ComfyUI's
  `detect_model_quantization`) and switches to a dedicated load path:
  - skips `torch_dtype` so the bundled BitsAndBytes config drives dtype;
  - forces `attn_implementation="sdpa"` (BnB linear layers are not
    compatible with `flash_attention_2`);
  - keeps `device_map="cuda"`.
  `bitsandbytes` is required and is checked at load time with a clear
  error message. Set `VIBEVOICE_MODEL_PATH=FabioSarracino/VibeVoice-Large-Q8`
  to match ComfyUI's `VibeVoice-Large-Q8` workflow.

## [0.5.3] — 2026-04-28

### Fixed

- **Stopped pre-processing reference voice clips by default.** Earlier
  versions trimmed leading/trailing silence with
  `librosa.effects.trim(top_db=30)` and hard-capped the clip to 10 s before
  feeding it to the processor. The ComfyUI VibeVoice front-end and
  Microsoft's reference inference demo do **neither** — they pass the raw
  resampled mono clip straight in and rely on the processor's own
  `AudioNormalizer` (-25 dBFS RMS) for level-matching. With our
  preprocessing on:
  - silence trim could shave off soft phoneme boundaries (consonants,
    fricatives) → audible artefacts at the start of segments;
  - 10 s cap discarded phoneme variety the model uses to generalise the
    voice → degraded pronunciation/accent on non-English voices.
  New defaults (env-overridable):
  - `VOICE_SAMPLE_TRIM_SILENCE=false` (was `true`)
  - `VOICE_SAMPLE_MAX_DURATION=0` — `0` disables the cap (was `10`).
  `prepare_voice_sample` now treats `max_duration=0` as "no cap".

## [0.5.2] — 2026-04-28

### Fixed

- **Removed noise-scheduler override** in `TTSService.load_model`. Earlier
  versions reconstructed the model's `noise_scheduler` after loading with
  `algorithm_type='sde-dpmsolver++'` and `beta_schedule='squaredcos_cap_v2'`.
  This had two audible side-effects vs ComfyUI / Microsoft's reference
  inference demo (which both leave the scheduler at the model's defaults):
  - The SDE variant of dpmsolver++ injects fresh Gaussian noise at every
    reverse-diffusion step → audible as bleeding / strange noises at the
    start of generated segments.
  - `squaredcos_cap_v2` overrode the β-schedule the diffusion head was
    actually trained with (`config.diffusion_head_config.ddpm_beta_schedule`)
    → distorted diffusion trajectory, hurting pronunciation/accent fidelity
    (especially noticeable on non-English voices).
  Generation now uses the model's own `DPMSolverMultistepScheduler`
  (deterministic `dpmsolver++` with the trained β-schedule), matching the
  ComfyUI implementation exactly.

## [0.5.1] — 2026-04-27

### Added

- **DEBUG-level generation trace** in `TTSService.generate_speech`. When
  `LOG_LEVEL=DEBUG`, every request emits a single block listing the
  exact `model.generate()` parameters (`cfg_scale`, effective
  `inference_steps`, `seed`, `do_sample`, `temperature`, `top_p`,
  `max_words_per_chunk`, `chunk_silence_ms`, `num_chunks`,
  `refresh_negative`, `return_speech`) plus per-speaker reference-clip
  stats (samples, duration, sample-rate, peak, RMS, dtype, shape) and a
  text preview. Intended for diagnosing quality regressions vs other
  VibeVoice front-ends (e.g. ComfyUI).
- New `voice_sources: Optional[List[str]]` arg on
  `TTSService.generate_speech`. Both routers now thread human-readable
  source identifiers through:
  - `/v1/vibevoice/generate` — `preset=<key> path=<file>` or
    `base64:<N>B` per speaker.
  - `/v1/audio/speech` — `voice=<req> resolved=<key> language=<lang>
    path=<file>`.

## [0.5.0] — 2026-04-27

### Changed (breaking)

- **Voice presets are now organized by language folder.** Files live under
  `<voices_dir>/<lang>/<name>.<ext>` (ISO 639-1 code), and the preset key
  used everywhere in the API is `<lang>/<name>` (e.g. `pl/Alice`). Files
  sitting at the root of `<voices_dir>` continue to load under their bare
  stem for backward compatibility (legacy demo voices).
- **`POST /v1/vibevoice/voices` requires a `language` form field** —
  uploads must declare the language so the file is stored in the correct
  subfolder and registered as `<language>/<name>`. Returns `409` if a
  preset with the same `<language>/<name>` already exists.
- **`DELETE /v1/vibevoice/voices/{voice_name}` is now
  `DELETE /v1/vibevoice/voices/{language}/{name}`** — language is a path
  segment so slashes don't break routing.
- **Updated generation defaults** (overridable via env / per-request):
  - `default_voice="man_2_pl"`, `default_language="pl"`
  - `default_cfg_scale=1.85` (was `1.3`)
  - `vibevoice_inference_steps=25` (was `10`)
  - `default_seed=0`, `default_do_sample=True` (was `False`)
  - `default_temperature=0.95` (was `1.0`), `default_top_p=0.95` (was `1.0`)
  - `default_max_words_per_chunk=100` (was `250`)
  - `default_chunk_silence_ms=500` (was `0`)
  These are also reflected in the Swagger-visible defaults on
  `OpenAITTSRequest` and `VibeVoiceGenerateRequest`. The `voice` field on
  `OpenAITTSRequest` is no longer required and defaults to `man_2_pl`.

### Added

- **`GET /v1/vibevoice/voices?language=<code>`** — optional ISO 639-1
  filter; returns only presets stored under the matching language folder.
  Same filter is also available on the OpenAI-compatible
  `GET /v1/audio/voices?language=<code>`.
- **Typed response models** — `VoiceInfo`, `VoiceListResponse`,
  `VoiceUploadResponse`, `ReloadVoicesResponse` (each entry now carries
  both a human-readable `language` and the `language_code`).
- **Richer Swagger annotations** on every voice endpoint: summaries,
  response descriptions, error response models (`400/404/409/503`), and
  documented path/query/form parameters with examples.

### Fixed

- Voice keys are now unambiguous: language-folder voices always register
  as `<lang>/<stem>` instead of relying on collision-detection fallbacks.

## [0.4.1] — 2026-04-26

### Added

- **OpenAI-compatible `POST /v1/audio/speech` parity** — the endpoint now
  accepts the same VibeVoice extension fields as `/v1/vibevoice/generate`:
  `cfg_scale`, `inference_steps`, `seed`, `do_sample`, `temperature`,
  `top_p`, `max_words_per_chunk`, `chunk_silence_ms`. Stock OpenAI clients
  that only send the standard fields are unaffected and pick up the server
  defaults automatically.

### Changed

- Richer FastAPI/OpenAPI annotations across the TTS request schemas and
  endpoints. Every field on `OpenAITTSRequest`, `VibeVoiceGenerateRequest`,
  and `SpeakerConfig` now carries a precise description (including
  `inference_steps` being clarified as the DDPM diffusion-step count, a.k.a.
  `diffusion_steps` in some VibeVoice front-ends). Both `/v1/vibevoice/generate`
  and `/v1/audio/speech` have detailed operation summaries and docstrings,
  visible in `/docs` and the generated OpenAPI spec.

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
