"""OpenAI-compatible TTS endpoint."""

import logging
import time
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse

from api.models import OpenAITTSRequest, ErrorResponse
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import audio_to_bytes, get_content_type, get_audio_duration
from api.utils.streaming import create_streaming_response
from api.config import settings

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/audio", tags=["OpenAI Compatible"])

# Global service instances (initialized in main.py)
tts_service: TTSService = None
voice_manager: VoiceManager = None


def get_tts_service() -> TTSService:
    """Dependency to get TTS service."""
    if tts_service is None or not tts_service.is_loaded:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    return tts_service


def get_voice_manager() -> VoiceManager:
    """Dependency to get voice manager."""
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


@router.post(
    "/speech",
    summary="Generate speech (OpenAI-compatible)",
    response_description="Encoded audio bytes in the requested `response_format`.",
    responses={
        200: {
            "content": {
                "audio/mpeg": {}, "audio/wav": {}, "audio/flac": {},
                "audio/ogg": {}, "audio/aac": {}, "audio/mp4": {},
            },
            "description": "Generated audio.",
        },
        400: {"model": ErrorResponse, "description": "Unknown voice or invalid params."},
        503: {"model": ErrorResponse, "description": "TTS service not ready."},
    },
)
async def create_speech(
    request: OpenAITTSRequest,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """OpenAI-compatible Text-to-Speech endpoint, backed by VibeVoice.

    Drop-in replacement for `POST /v1/audio/speech` from the OpenAI API:
    accepts the standard `model`, `input`, `voice`, `response_format`, and
    `speed` fields. The voice is resolved against OpenAI voice names
    (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) **or** any
    on-disk VibeVoice preset name.

    **Extensions** (optional, ignored by stock OpenAI clients):

    - `language` — ISO 639-1 code; resolves the voice under
      `<voices_dir>/<language>/` for accent control.
    - `cfg_scale`, `inference_steps` — diffusion controls.
    - `do_sample`, `temperature`, `top_p`, `seed` — LLM sampling controls.
    - `max_words_per_chunk`, `chunk_silence_ms` — sentence-aware chunking
      for long inputs (long generations otherwise lose quality).

    All extension fields fall back to server-side defaults
    (`DEFAULT_*` env vars) when omitted.
    """
    try:
        voice_audio = None
        resolved_voice = request.voice
        is_openai_voice = request.voice in voices.OPENAI_VOICE_MAPPING

        # If a language is supplied, prefer a preset stored under <voices_dir>/<language>/
        if request.language:
            language_key = voices.resolve_voice_for_language(
                request.voice,
                request.language,
                is_openai_voice=is_openai_voice,
            )
            if language_key:
                resolved_voice = language_key
                voice_audio = voices.load_voice_audio(language_key, is_openai_voice=False)
            else:
                available_in_lang = sorted(
                    k for k, p in voices.voice_presets.items()
                    if f"/{request.language.lower().strip()}/" in f"/{p.lower()}/"
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No voice matching '{request.voice}' found for language "
                        f"'{request.language}'. Presets available in that language: "
                        f"{', '.join(available_in_lang) if available_in_lang else '(none)'}"
                    ),
                )

        # Fallback: try as OpenAI voice, then as direct VibeVoice preset name
        if voice_audio is None:
            voice_audio = voices.load_voice_audio(request.voice, is_openai_voice=True)
        if voice_audio is None:
            voice_audio = voices.load_voice_audio(request.voice, is_openai_voice=False)

        if voice_audio is None:
            available_openai = ', '.join(voices.OPENAI_VOICE_MAPPING.keys())
            available_presets = ', '.join(sorted(voices.voice_presets.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not found. OpenAI voices: {available_openai}. VibeVoice presets: {available_presets}"
            )
        
        # Format text as single-speaker script
        formatted_script = tts.format_script_for_single_speaker(request.input, speaker_id=0)
        
        # Generate speech with timing
        # Note: OpenAI API doesn't support streaming in the same way, but we can use chunked transfer
        start_time = time.time()
        # Resolve params with server defaults (request overrides win)
        cfg_scale = request.cfg_scale if request.cfg_scale is not None else settings.default_cfg_scale
        inference_steps = request.inference_steps  # None -> service uses model default
        max_words = (
            request.max_words_per_chunk if request.max_words_per_chunk is not None
            else settings.default_max_words_per_chunk
        )
        chunk_silence_ms = (
            request.chunk_silence_ms if request.chunk_silence_ms is not None
            else settings.default_chunk_silence_ms
        )
        do_sample = request.do_sample
        if do_sample is None:
            do_sample = (
                request.temperature is not None
                or request.top_p is not None
                or settings.default_do_sample
            )
        temperature = (
            request.temperature if request.temperature is not None
            else (settings.default_temperature if do_sample else None)
        )
        top_p = (
            request.top_p if request.top_p is not None
            else (settings.default_top_p if do_sample else None)
        )

        audio = tts.generate_speech(
            text=formatted_script,
            voice_samples=[voice_audio],
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            seed=request.seed,
            stream=False,  # For OpenAI compatibility, generate all at once
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_words_per_chunk=max_words,
            chunk_silence_ms=chunk_silence_ms,
        )
        generation_time = time.time() - start_time
        
        # Calculate audio duration
        audio_duration = get_audio_duration(audio, sample_rate=24000)
        
        # Log generation details at INFO level
        text_preview = request.input[:100] + "..." if len(request.input) > 100 else request.input
        logger.info(
            f"Generated speech - Text: {text_preview} | Voice: {request.voice} "
            f"(resolved: {resolved_voice}) | Language: {request.language or 'auto'} | "
            f"Model: {request.model} ({settings.vibevoice_model_path}) | "
            f"CFG: {cfg_scale} | Steps: {inference_steps if inference_steps is not None else settings.vibevoice_inference_steps} | "
            f"Sample: {bool(do_sample)} | Temp: {temperature} | TopP: {top_p} | "
            f"MaxWords/Chunk: {max_words} | Seed: {request.seed if request.seed is not None else 'None'} | "
            f"Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
        )
        
        # Convert to requested format
        audio_bytes = audio_to_bytes(
            audio,
            sample_rate=24000,
            format=request.response_format
        )
        
        # Return audio response
        return Response(
            content=audio_bytes,
            media_type=get_content_type(request.response_format),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/voices",
    summary="List voices (OpenAI-compatible)",
    response_description="OpenAI-shaped voice listing (`{object: 'list', data: [...]}`).",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid `language` filter."},
    },
)
async def list_voices(
    language: str | None = Query(
        None,
        description="Optional ISO 639-1 language filter (e.g. `pl`, `en`). When set, only presets stored under `<voices_dir>/<language>/` are returned, and the OpenAI standard voice aliases are omitted.",
        examples=["pl"],
        min_length=2,
        max_length=3,
    ),
    voices: VoiceManager = Depends(get_voice_manager),
):
    """List all available voices in OpenAI-compatible format.

    Without `language`: includes the standard OpenAI voice aliases
    (`alloy`, `echo`, ...) when their mapped presets exist, plus every
    custom preset registered on the server.

    With `language`: returns only the presets stored under that language
    folder. The OpenAI aliases are intentionally omitted because they map
    to a fixed set of (typically English) presets.
    """
    try:
        voice_list = []

        if language is None:
            for openai_name, vibevoice_preset in voices.OPENAI_VOICE_MAPPING.items():
                if vibevoice_preset in voices.voice_presets:
                    voice_list.append({
                        "id": openai_name,
                        "object": "voice",
                        "name": openai_name,
                    })

            all_voices = voices.list_available_voices()
            for voice in all_voices:
                if voice["name"] not in voices.OPENAI_VOICE_MAPPING.values():
                    voice_list.append({
                        "id": voice["name"],
                        "object": "voice",
                        "name": voice["name"],
                        "language": voice["language"],
                        "language_code": voice["language_code"],
                    })
        else:
            for voice in voices.list_available_voices(language=language):
                voice_list.append({
                    "id": voice["name"],
                    "object": "voice",
                    "name": voice["name"],
                    "language": voice["language"],
                    "language_code": voice["language_code"],
                })

        return {
            "object": "list",
            "data": voice_list,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

