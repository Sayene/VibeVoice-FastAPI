"""Configuration management for VibeVoice API."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model Configuration
    vibevoice_model_path: str = Field(
        default="microsoft/VibeVoice-1.5B",
        description="Path to VibeVoice model weights (local path or HuggingFace model ID)"
    )
    vibevoice_device: str = Field(
        default="cuda",
        description="Device for inference: cuda, cpu, or mps"
    )
    vibevoice_inference_steps: int = Field(
        default=25,
        description="Number of diffusion inference steps"
    )
    vibevoice_dtype: Optional[str] = Field(
        default=None,
        description="Model dtype: bfloat16, float16, or float32 (auto-detected if None)"
    )
    vibevoice_attn_implementation: Optional[str] = Field(
        default=None,
        description="Attention implementation: flash_attention_2, sdpa, or eager (auto-detected if None)"
    )
    torch_compile: bool = Field(
        default=False,
        description="Enable torch.compile for optimized inference (20-50% speedup, but first request is slower due to compilation)"
    )
    vibevoice_quantization: Optional[str] = Field(
        default=None,
        description="Quantization method: 'int8_torchao', 'int4_torchao', or None for full precision"
    )
    torch_compile_mode: str = Field(
        default="default",
        description="torch.compile mode: 'default', 'reduce-overhead', or 'max-autotune' (slower compile, faster inference)"
    )

    # Voice sample pre-processing
    voice_sample_max_duration: float = Field(
        default=0.0,
        description=(
            "Hard cap (in seconds) on the reference voice clip fed to the model. "
            "0 disables the cap (default), matching the ComfyUI VibeVoice front-end "
            "and Microsoft's reference inference demo. Set a positive value only if "
            "your reference clips are unusually long."
        )
    )
    voice_sample_trim_silence: bool = Field(
        default=False,
        description=(
            "If true, strip leading/trailing silence from reference voice clips "
            "with librosa.effects.trim before feeding them to the model. Disabled "
            "by default — trimming can shave off soft phoneme boundaries and hurt "
            "pronunciation/accent fidelity on non-English voices. The processor "
            "already does its own RMS-based dB normalisation."
        )
    )
    voice_sample_trim_db: float = Field(
        default=30.0,
        description="Silence threshold in dB for librosa.effects.trim."
    )

    # Voice Configuration
    voices_dir: str = Field(
        default="/app/voices",  # Docker default; override with VOICES_DIR=demo/voices for local dev
        description="Directory containing voice preset audio files"
    )
    openai_voice_mapping: str = Field(
        default='{"alloy": "en-Alice_woman", "echo": "en-Carter_man", "fable": "en-Maya_woman", "onyx": "en-Frank_man", "nova": "en-Mary_woman_bgm", "shimmer": "en-Alice_woman"}',
        description="JSON mapping of OpenAI voice names to VibeVoice preset names"
    )
    
    # API Server Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8001,
        description="API server port"
    )
    api_workers: int = Field(
        default=1,
        description="Number of API workers (keep at 1 for model loading)"
    )
    api_cors_origins: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated)"
    )
    
    # Generation Defaults
    default_voice: str = Field(
        default="man_2_pl",
        description="Default voice preset key when none is supplied on the request."
    )
    default_language: str = Field(
        default="pl",
        description="Default ISO 639-1 language code when none is supplied on the request."
    )
    default_speed: float = Field(
        default=1.0,
        description="Default playback speed multiplier (0.25–4.0)."
    )
    default_cfg_scale: float = Field(
        default=1.85,
        description="Default CFG scale for generation (1.0-2.0, higher = more faithful to prompt)"
    )
    default_response_format: str = Field(
        default="mp3",
        description="Default audio response format"
    )
    default_seed: int = Field(
        default=0,
        description="Default RNG seed when none is supplied on the request."
    )
    max_generation_length: int = Field(
        default=90 * 60,  # 90 minutes in seconds
        description="Maximum generation length in seconds"
    )
    default_do_sample: bool = Field(
        default=True,
        description="Whether to use sampling for text generation (False = greedy decoding)"
    )
    default_temperature: float = Field(
        default=0.95,
        description="Temperature for sampling (only used if do_sample=True)"
    )
    default_top_p: float = Field(
        default=0.95,
        description="Top-p (nucleus) sampling (only used if do_sample=True)"
    )
    default_top_k: int = Field(
        default=50,
        description="Top-k sampling (only used if do_sample=True)"
    )
    default_repetition_penalty: float = Field(
        default=1.0,
        description="Repetition penalty (1.0 = no penalty)"
    )
    default_max_words_per_chunk: int = Field(
        default=100,
        description=(
            "Default max words per generated chunk. Long scripts are split at "
            "sentence boundaries and synthesized sequentially to preserve quality. "
            "Set 0 to disable chunking."
        ),
    )
    default_chunk_silence_ms: int = Field(
        default=500,
        description="Silence (ms) inserted between concatenated chunks (non-streaming).",
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive)"
    )
    
    @property
    def normalized_log_level(self) -> str:
        """Get log level normalized to uppercase for logging module."""
        return self.log_level.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.api_cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.api_cors_origins.split(",")]
    
    def get_device(self) -> str:
        """Get the appropriate device, checking availability."""
        import torch
        
        if self.vibevoice_device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif self.vibevoice_device == "mps" and not torch.backends.mps.is_available():
            print("WARNING: MPS requested but not available, falling back to CPU")
            return "cpu"
        return self.vibevoice_device
    
    def get_dtype(self):
        """Get the appropriate dtype for the device."""
        import torch
        
        if self.vibevoice_dtype:
            return getattr(torch, self.vibevoice_dtype)
        
        device = self.get_device()
        if device == "cuda":
            return torch.bfloat16
        elif device == "mps":
            return torch.float32
        else:
            return torch.float32
    
    def get_attn_implementation(self) -> str:
        """Get the appropriate attention implementation."""
        if self.vibevoice_attn_implementation:
            return self.vibevoice_attn_implementation
        
        device = self.get_device()
        if device == "cuda":
            # Try flash_attention_2 first, fallback to sdpa
            try:
                import flash_attn
                return "flash_attention_2"
            except ImportError:
                return "sdpa"
        else:
            return "sdpa"


# Global settings instance
settings = Settings()


