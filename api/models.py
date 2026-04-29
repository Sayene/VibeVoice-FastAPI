"""Pydantic models for API request/response schemas."""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field, validator


# ============================================================
# OpenAI-Compatible TTS Models
# ============================================================

class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request schema."""
    
    model: str = Field(
        default="tts-1",
        description="Model to use: tts-1 or tts-1-hd (both map to VibeVoice)"
    )
    input: str = Field(
        ...,
        description="The text to generate audio for",
        max_length=4096
    )
    voice: str = Field(
        default="man_2_pl",
        description="Voice to use: OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) or any VibeVoice preset key (e.g. `pl/Alice`, `man_2_pl`)."
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"]] = Field(
        default="mp3",
        description="Audio format for the response"
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of generated audio (0.25 to 4.0)"
    )
    language: Optional[str] = Field(
        default="pl",
        description="ISO 639-1 language code (e.g. 'en', 'pl', 'de'). When set, the voice is resolved to a preset stored under <voices_dir>/<language>/."
    )

    # ---- VibeVoice extensions (non-OpenAI) ---------------------------------
    # These are accepted for parity with /v1/vibevoice/generate. Any client
    # using only the standard OpenAI fields will continue to work unchanged.

    cfg_scale: Optional[float] = Field(
        default=1.85,
        ge=1.0,
        le=2.0,
        description="Classifier-free guidance scale (1.0–2.0). Higher = follows the script more strictly.",
    )
    inference_steps: Optional[int] = Field(
        default=25,
        ge=5,
        le=50,
        description="Number of DDPM diffusion steps (5–50). Higher = better fidelity, slower.",
    )
    seed: Optional[int] = Field(
        default=0,
        description="Random seed for reproducible generation.",
    )
    do_sample: Optional[bool] = Field(
        default=False,
        description="Use sampling instead of greedy decoding. Auto-enabled when `temperature` or `top_p` is supplied. False (default) matches ComfyUI's `use_sampling=false`.",
    )
    temperature: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=5.0,
        description="Sampling temperature (only used when do_sample=True).",
    )
    top_p: Optional[float] = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling top_p (only used when do_sample=True).",
    )
    max_words_per_chunk: Optional[int] = Field(
        default=100,
        ge=0,
        le=2000,
        description=(
            "Max words per generated chunk; long inputs are split at sentence "
            "boundaries and synthesized sequentially. 0 disables chunking."
        ),
    )
    chunk_silence_ms: Optional[int] = Field(
        default=500,
        ge=0,
        le=5000,
        description="Silence (ms) inserted between concatenated chunks.",
    )
    warmup_text: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "Short string injected right after the `Speaker N:` label of every "
            "chunk before generation, to suppress start-of-audio artefacts. "
            "Empty string disables; `null` falls back to the server default "
            "(`DEFAULT_WARMUP_TEXT`)."
        ),
    )

    # Note: Voice validation removed to allow any VibeVoice preset name
    # Validation happens in the endpoint with proper error messages


# ============================================================
# VibeVoice-Specific Models
# ============================================================

class SpeakerConfig(BaseModel):
    """Configuration for a single speaker referenced in the script."""

    speaker_id: int = Field(
        ...,
        ge=0,
        le=3,
        description="Speaker ID matching `Speaker N:` labels in the script (0-3).",
    )
    voice_preset: Optional[str] = Field(
        default=None,
        description=(
            "Preset key (see `GET /v1/vibevoice/voices`). For voices organized by "
            "language, this is `<lang>/<name>` (e.g. `pl/Alice`); legacy root-level "
            "demo voices use the bare stem (e.g. `en-Alice_woman`). "
            "Mutually exclusive with `voice_sample_base64`."
        ),
        examples=["pl/Alice", "en-Alice_woman"],
    )
    voice_sample_base64: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded reference audio (wav/mp3/flac/ogg/m4a) for "
            "zero-shot voice cloning. Resampled to 24 kHz mono and trimmed "
            "to ~10s server-side. Mutually exclusive with `voice_preset`."
        ),
    )
    
    @validator("voice_preset", "voice_sample_base64")
    def validate_voice_source(cls, v, values):
        """Ensure at least one voice source is provided."""
        if "voice_preset" in values and not values.get("voice_preset") and not v:
            raise ValueError("Either voice_preset or voice_sample_base64 must be provided")
        return v


class VibeVoiceGenerateRequest(BaseModel):
    """VibeVoice-specific generation request with multi-speaker support.

    Supports up to 4 distinct speakers, custom voice cloning via base64 or
    on-disk presets, sentence-aware chunking for long scripts, and full
    control over the diffusion + sampling parameters.
    """

    script: str = Field(
        ...,
        description=(
            "Multi-speaker script. Each turn must start with a `Speaker N:` "
            "label (N = 0..3), e.g. `Speaker 0: Hello.\\nSpeaker 1: Hi there.`. "
            "Plain text without labels is treated as a single Speaker 0 turn."
        ),
        max_length=100000,
        examples=["Speaker 0: Hello there.\nSpeaker 1: Hi, how are you?"],
    )
    speakers: List[SpeakerConfig] = Field(
        ...,
        min_items=1,
        max_items=4,
        description=(
            "Speaker configurations, one per speaker_id referenced in the script. "
            "IDs must be sequential starting at 0. Each speaker must supply either "
            "`voice_preset` (an on-disk preset name) or `voice_sample_base64` "
            "(inline reference clip)."
        ),
    )
    cfg_scale: Optional[float] = Field(
        default=1.85,
        ge=1.0,
        le=2.0,
        description=(
            "Classifier-free guidance scale. Higher values follow the script "
            "more strictly at the cost of naturalness. Typical range 1.2–1.9."
        ),
    )
    inference_steps: Optional[int] = Field(
        default=25,
        ge=5,
        le=50,
        description=(
            "Number of DDPM diffusion denoising steps (a.k.a. `diffusion_steps` "
            "in some VibeVoice front-ends). More steps = higher fidelity but "
            "slower generation. 10 is a good default; 20–30 for top quality."
        ),
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"]] = Field(
        default="mp3",
        description="Audio container/codec for the response body.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description=(
            "If true, the response is streamed in real time over Server-Sent "
            "Events. When chunking is active, chunks are streamed sequentially "
            "through the same SSE response."
        ),
    )
    seed: Optional[int] = Field(
        default=0,
        description="Random seed for reproducible generation. Set to a different value for non-deterministic output.",
    )
    do_sample: Optional[bool] = Field(
        default=False,
        description=(
            "If true, use stochastic sampling for the LLM; if false (default), "
            "use greedy decoding. Auto-enabled when `temperature` or `top_p` is "
            "supplied. False matches ComfyUI's `use_sampling=false`."
        ),
    )
    temperature: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=5.0,
        description=(
            "Sampling temperature for the LLM. Only applied when `do_sample` "
            "is true (or auto-enabled). Lower = more deterministic, higher = "
            "more varied. Typical range 0.7–1.2."
        ),
    )
    top_p: Optional[float] = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Nucleus (top-p) sampling threshold for the LLM. Only applied when "
            "`do_sample` is true (or auto-enabled). Typical 0.9–0.95."
        ),
    )
    max_words_per_chunk: Optional[int] = Field(
        default=100,
        ge=0,
        le=2000,
        description=(
            "Maximum words per generated chunk. Long scripts are split at "
            "sentence boundaries (sub-split at `,`/`;` for very long sentences) "
            "and synthesized sequentially to preserve quality, since VibeVoice "
            "degrades on very long single-shot generations. Multi-speaker "
            "scripts are chunked per-turn — chunks never cross speaker "
            "boundaries and `Speaker N:` labels are re-emitted in each chunk. "
            "Set `0` to disable chunking."
        ),
    )
    chunk_silence_ms: Optional[int] = Field(
        default=500,
        ge=0,
        le=5000,
        description=(
            "Silence (in milliseconds) inserted between concatenated chunks. "
            "Only applies in non-streaming mode."
        ),
    )
    warmup_text: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "Short string injected right after the `Speaker N:` label of every "
            "chunk before generation, to suppress start-of-audio artefacts. "
            "Empty string disables; `null` falls back to the server default "
            "(`DEFAULT_WARMUP_TEXT`)."
        ),
    )

    @validator("speakers")
    def validate_speaker_ids(cls, v):
        """Ensure speaker IDs are sequential starting from 0."""
        speaker_ids = sorted([s.speaker_id for s in v])
        expected_ids = list(range(len(v)))
        if speaker_ids != expected_ids:
            raise ValueError(f"Speaker IDs must be sequential starting from 0. Got: {speaker_ids}")
        return v


class VibeVoiceGenerateResponse(BaseModel):
    """Response for VibeVoice generation."""
    
    audio_url: Optional[str] = Field(
        default=None,
        description="URL to download the generated audio (for non-streaming)"
    )
    duration: Optional[float] = Field(
        default=None,
        description="Duration of generated audio in seconds"
    )
    format: str = Field(
        ...,
        description="Audio format of the response"
    )
    sample_rate: int = Field(
        default=24000,
        description="Sample rate of the audio"
    )


# ============================================================
# Common Models
# ============================================================

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: dict = Field(
        ...,
        description="Error details"
    )
    
    @staticmethod
    def from_exception(exc: Exception, status_code: int = 500) -> "ErrorResponse":
        """Create error response from exception."""
        return ErrorResponse(
            error={
                "message": str(exc),
                "type": type(exc).__name__,
                "code": status_code
            }
        )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        default="healthy",
        description="Service health status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    device: str = Field(
        ...,
        description="Device being used for inference"
    )
    model_path: str = Field(
        ...,
        description="Path to the loaded model"
    )


class VoiceInfo(BaseModel):
    """Single voice preset entry."""

    name: str = Field(
        ...,
        description="Preset key. Language-folder voices are keyed `<lang>/<stem>` (e.g. `pl/Alice`); legacy root-level files use the bare stem.",
        examples=["pl/Alice", "en/Carter_man"],
    )
    path: str = Field(
        ...,
        description="Absolute or working-directory-relative path to the voice file on disk.",
    )
    language: str = Field(
        ...,
        description="Human-readable language name (e.g. `Polish`, `English`, or `Unknown` if it could not be inferred).",
    )
    language_code: str = Field(
        default="",
        description="ISO 639-1 code (e.g. `pl`, `en`). Empty string when the language could not be inferred.",
    )


class VoiceUploadResponse(VoiceInfo):
    """Response from uploading a new voice preset."""
    pass


class ReloadVoicesResponse(BaseModel):
    """Response from rescanning the voices directory."""

    count: int = Field(..., description="Total number of presets registered after reload.")
    voices: List[str] = Field(..., description="Sorted list of all preset keys.")


class VoiceListResponse(BaseModel):
    """Response listing available voices."""

    voices: List[VoiceInfo] = Field(
        ...,
        description="Available voice presets. When `language` was supplied as a query filter, only presets stored under that language folder are included.",
    )

