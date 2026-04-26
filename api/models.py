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
        ...,
        description="Voice to use: OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) or any VibeVoice preset name"
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
        default=None,
        description="ISO 639-1 language code (e.g. 'en', 'pl', 'de'). When set, the voice is resolved to a preset stored under <voices_dir>/<language>/."
    )

    # ---- VibeVoice extensions (non-OpenAI) ---------------------------------
    # These are accepted for parity with /v1/vibevoice/generate. Any client
    # using only the standard OpenAI fields will continue to work unchanged.

    cfg_scale: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=2.0,
        description=(
            "Classifier-free guidance scale (1.0–2.0). Defaults to "
            "server `DEFAULT_CFG_SCALE` (typically 1.3)."
        ),
    )
    inference_steps: Optional[int] = Field(
        default=None,
        ge=5,
        le=50,
        description=(
            "Number of DDPM diffusion steps (5–50). Higher = better fidelity, "
            "slower. Defaults to server `VIBEVOICE_INFERENCE_STEPS`."
        ),
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )
    do_sample: Optional[bool] = Field(
        default=None,
        description=(
            "Use sampling instead of greedy decoding. Auto-enabled when "
            "`temperature` or `top_p` is supplied. Defaults to server "
            "`DEFAULT_DO_SAMPLE`."
        ),
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Sampling temperature (only used when do_sample=True).",
    )
    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling top_p (only used when do_sample=True).",
    )
    max_words_per_chunk: Optional[int] = Field(
        default=None,
        ge=0,
        le=2000,
        description=(
            "Max words per generated chunk; long inputs are split at sentence "
            "boundaries and synthesized sequentially. 0 disables chunking. "
            "Defaults to server `DEFAULT_MAX_WORDS_PER_CHUNK`."
        ),
    )
    chunk_silence_ms: Optional[int] = Field(
        default=None,
        ge=0,
        le=5000,
        description=(
            "Silence (ms) inserted between concatenated chunks. Defaults to "
            "server `DEFAULT_CHUNK_SILENCE_MS`."
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
            "Name of an on-disk voice preset (see `GET /v1/vibevoice/voices`). "
            "Mutually exclusive with `voice_sample_base64`."
        ),
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
        default=1.3,
        ge=1.0,
        le=2.0,
        description=(
            "Classifier-free guidance scale. Higher values follow the script "
            "more strictly at the cost of naturalness. Typical range 1.2–1.5; "
            "1.3 is the model default."
        ),
    )
    inference_steps: Optional[int] = Field(
        default=10,
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
        default=None,
        description="Random seed for reproducible generation. Omit for non-deterministic output.",
    )
    do_sample: Optional[bool] = Field(
        default=None,
        description=(
            "If true, use stochastic sampling for the LLM; if false, use greedy "
            "decoding (more stable, recommended default). Auto-enabled when "
            "`temperature` or `top_p` is supplied."
        ),
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description=(
            "Sampling temperature for the LLM. Only applied when `do_sample` "
            "is true (or auto-enabled). Lower = more deterministic, higher = "
            "more varied. Typical range 0.7–1.2."
        ),
    )
    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Nucleus (top-p) sampling threshold for the LLM. Only applied when "
            "`do_sample` is true (or auto-enabled). Typical 0.9–0.95."
        ),
    )
    max_words_per_chunk: Optional[int] = Field(
        default=None,
        ge=0,
        le=2000,
        description=(
            "Maximum words per generated chunk. Long scripts are split at "
            "sentence boundaries (sub-split at `,`/`;` for very long sentences) "
            "and synthesized sequentially to preserve quality, since VibeVoice "
            "degrades on very long single-shot generations. Multi-speaker "
            "scripts are chunked per-turn — chunks never cross speaker "
            "boundaries and `Speaker N:` labels are re-emitted in each chunk. "
            "Set `0` to disable chunking. Omit to use the server default "
            "(`DEFAULT_MAX_WORDS_PER_CHUNK`, typically 250)."
        ),
    )
    chunk_silence_ms: Optional[int] = Field(
        default=None,
        ge=0,
        le=5000,
        description=(
            "Silence (in milliseconds) inserted between concatenated chunks. "
            "Only applies in non-streaming mode. Omit to use the server "
            "default (`DEFAULT_CHUNK_SILENCE_MS`, typically 0)."
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


class VoiceListResponse(BaseModel):
    """Response listing available voices."""
    
    voices: List[dict] = Field(
        ...,
        description="List of available voice presets"
    )

