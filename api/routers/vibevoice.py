"""VibeVoice-specific TTS endpoints with multi-speaker support."""

import base64
import logging
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query, Path as PathParam
from fastapi.responses import Response

from api.models import (
    VibeVoiceGenerateRequest,
    VibeVoiceGenerateResponse,
    VoiceListResponse,
    VoiceUploadResponse,
    VoiceInfo,
    ReloadVoicesResponse,
    HealthResponse,
    ErrorResponse,
)
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager, prepare_voice_sample
from api.utils.audio_utils import audio_to_bytes, get_audio_duration
from api.utils.streaming import create_streaming_response
from api.config import settings

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/vibevoice", tags=["VibeVoice Extended"])

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
    "/generate",
    summary="Generate multi-speaker speech",
    response_description="Audio bytes in the requested `response_format`, or an SSE stream when `stream=true`.",
    responses={
        200: {
            "content": {
                "audio/mpeg": {}, "audio/wav": {}, "audio/flac": {},
                "audio/ogg": {}, "audio/aac": {}, "audio/mp4": {},
                "text/event-stream": {},
            },
            "description": "Generated audio (or SSE stream).",
        },
        400: {"model": ErrorResponse, "description": "Invalid speaker config or unknown voice preset."},
        503: {"model": ErrorResponse, "description": "TTS service not ready."},
    },
)
async def generate_speech(
    request: VibeVoiceGenerateRequest,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """Generate speech from a multi-speaker script using VibeVoice.

    **Features**

    - Up to 4 distinct speakers, each driven by an on-disk preset
      (`voice_preset`) or an inline base64 reference clip
      (`voice_sample_base64`).
    - Full control over the diffusion + sampling pipeline:
      `cfg_scale`, `inference_steps` (a.k.a. diffusion steps),
      `do_sample`, `temperature`, `top_p`, `seed`.
    - **Sentence-aware chunking** for long scripts via
      `max_words_per_chunk` — chunks never cross speaker turns; each
      chunk re-emits its `Speaker N:` label and is synthesized
      sequentially to preserve quality. Use `chunk_silence_ms` to
      insert silence between concatenated chunks (non-streaming only).
    - Real-time delivery via Server-Sent Events when `stream=true`.

    **Voice presets**

    Presets are organized by ISO 639-1 language code on disk
    (`<voices_dir>/<lang>/<name>.<ext>`) and referenced as `<lang>/<name>`
    (e.g. `pl/Alice`). Use `GET /v1/vibevoice/voices` to discover what is
    available.

    **Script format**

    ```
    Speaker 0: First turn of dialogue.
    Speaker 1: Second speaker's reply.
    Speaker 0: Back to the first speaker.
    ```

    Returns the encoded audio (`response_format`) plus
    `X-Audio-Duration`, `X-Audio-Format`, and `X-Audio-Sample-Rate`
    headers in non-streaming mode.
    """
    try:
        # Load voice samples for each speaker
        voice_samples = []

        for speaker_config in sorted(request.speakers, key=lambda s: s.speaker_id):
            if speaker_config.voice_sample_base64:
                # Decode base64 audio
                try:
                    audio_bytes = base64.b64decode(speaker_config.voice_sample_base64)
                    import io
                    import soundfile as sf
                    audio_data, sr = sf.read(io.BytesIO(audio_bytes))

                    if sr != 24000:
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)

                    if len(audio_data.shape) > 1:
                        import numpy as np
                        audio_data = np.mean(audio_data, axis=1)

                    audio_data = prepare_voice_sample(
                        audio_data.astype("float32"),
                        sample_rate=24000,
                        max_duration=settings.voice_sample_max_duration,
                        trim_silence=settings.voice_sample_trim_silence,
                        trim_db=settings.voice_sample_trim_db,
                    )
                    voice_samples.append(audio_data)

                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decode voice sample for speaker {speaker_config.speaker_id}: {str(e)}"
                    )

            elif speaker_config.voice_preset:
                audio_data = voices.load_voice_audio(speaker_config.voice_preset, is_openai_voice=False)

                if audio_data is None:
                    available_voices = [v["name"] for v in voices.list_available_voices()]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Voice preset '{speaker_config.voice_preset}' not found. Available: {', '.join(available_voices)}"
                    )

                voice_samples.append(audio_data)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker {speaker_config.speaker_id} must have either voice_preset or voice_sample_base64"
                )

        voice_list = []
        for speaker_config in sorted(request.speakers, key=lambda s: s.speaker_id):
            if speaker_config.voice_preset:
                voice_list.append(f"speaker{speaker_config.speaker_id}={speaker_config.voice_preset}")
            else:
                voice_list.append(f"speaker{speaker_config.speaker_id}=base64_audio")
        voices_str = ", ".join(voice_list)

        actual_inference_steps = request.inference_steps if request.inference_steps is not None else settings.vibevoice_inference_steps
        max_words = request.max_words_per_chunk if request.max_words_per_chunk is not None else settings.default_max_words_per_chunk
        chunk_silence_ms = request.chunk_silence_ms if request.chunk_silence_ms is not None else settings.default_chunk_silence_ms
        do_sample = request.do_sample
        if do_sample is None and (request.temperature is not None or request.top_p is not None):
            do_sample = True

        if request.stream:
            text_preview = request.script[:100] + "..." if len(request.script) > 100 else request.script
            logger.info(
                f"Generating speech (streaming) - Text: {text_preview} | Voices: {voices_str} | "
                f"Model: {settings.vibevoice_model_path} | CFG: {request.cfg_scale} | "
                f"Steps: {actual_inference_steps} | Seed: {request.seed if request.seed is not None else 'None'} | "
                f"Sample: {bool(do_sample)} | Temp: {request.temperature} | TopP: {request.top_p} | "
                f"MaxWords/Chunk: {max_words}"
            )

            audio_stream = tts.generate_speech(
                text=request.script,
                voice_samples=voice_samples,
                cfg_scale=request.cfg_scale,
                inference_steps=request.inference_steps,
                seed=request.seed,
                stream=True,
                do_sample=do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                max_words_per_chunk=max_words,
            )

            return create_streaming_response(
                audio_stream,
                format=request.response_format,
                sample_rate=24000,
                use_sse=True
            )

        else:
            start_time = time.time()
            audio = tts.generate_speech(
                text=request.script,
                voice_samples=voice_samples,
                cfg_scale=request.cfg_scale,
                inference_steps=request.inference_steps,
                seed=request.seed,
                stream=False,
                do_sample=do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                max_words_per_chunk=max_words,
                chunk_silence_ms=chunk_silence_ms,
            )
            generation_time = time.time() - start_time

            audio_duration = get_audio_duration(audio, sample_rate=24000)

            text_preview = request.script[:100] + "..." if len(request.script) > 100 else request.script
            logger.info(
                f"Generated speech - Text: {text_preview} | Voices: {voices_str} | "
                f"Model: {settings.vibevoice_model_path} | CFG: {request.cfg_scale} | "
                f"Steps: {actual_inference_steps} | Seed: {request.seed if request.seed is not None else 'None'} | "
                f"Sample: {bool(do_sample)} | Temp: {request.temperature} | TopP: {request.top_p} | "
                f"MaxWords/Chunk: {max_words} | "
                f"Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
            )

            audio_bytes = audio_to_bytes(
                audio,
                sample_rate=24000,
                format=request.response_format
            )

            duration = audio_duration

            from api.utils.audio_utils import get_content_type
            return Response(
                content=audio_bytes,
                media_type=get_content_type(request.response_format),
                headers={
                    "Content-Disposition": f"attachment; filename=vibevoice_output.{request.response_format}",
                    "X-Audio-Duration": str(duration),
                    "X-Audio-Format": request.response_format,
                    "X-Audio-Sample-Rate": "24000"
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
    response_model=VoiceListResponse,
    summary="List voice presets",
    response_description="Voice presets registered on the server.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid `language` filter."},
    },
)
async def list_voices(
    language: str | None = Query(
        None,
        description="Optional ISO 639-1 language filter (e.g. `pl`, `en`). When set, only presets stored under `<voices_dir>/<language>/` are returned.",
        examples=["pl"],
        min_length=2,
        max_length=3,
    ),
    voices: VoiceManager = Depends(get_voice_manager),
):
    """List all available VibeVoice presets, optionally filtered by language."""
    try:
        available_voices = voices.list_available_voices(language=language)
        return VoiceListResponse(voices=available_voices)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/voices",
    status_code=201,
    response_model=VoiceUploadResponse,
    summary="Upload a new voice preset",
    response_description="Metadata for the newly registered preset.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid filename, language, or undecodable audio."},
        409: {"model": ErrorResponse, "description": "A preset with the same name already exists for this language."},
    },
)
async def upload_voice(
    language: str = Form(
        ...,
        description="ISO 639-1 language code the voice belongs to (e.g. `pl`, `en`). The file is stored under `<voices_dir>/<language>/` and registered as `<language>/<name>`.",
        examples=["pl"],
        min_length=2,
        max_length=3,
    ),
    file: UploadFile = File(..., description="Audio file (wav/mp3/flac/ogg/m4a/aac)."),
    name: str | None = Form(
        None,
        description="Preset name (without extension). Defaults to the uploaded filename's stem.",
    ),
    voices: VoiceManager = Depends(get_voice_manager),
):
    """Upload a new voice preset for a specific language.

    The file is persisted under `<voices_dir>/<language>/<name>.<ext>` and
    registered as the preset key `<language>/<name>`. `language` must be a
    supported ISO 639-1 code (`en`, `pl`, `de`, `fr`, `es`, `it`, `pt`,
    `ru`, `ja`, `ko`, `zh`, `nl`, `cs`, `uk`, `tr`, `ar`, `hi`, `in`).
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix or ""
    preset_name = name or Path(filename).stem
    if not preset_name:
        raise HTTPException(status_code=400, detail="Could not determine voice name")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        info = voices.add_voice_from_bytes(preset_name, data, suffix, language=language)
    except ValueError as e:
        msg = str(e)
        status = 409 if "already exists" in msg else 400
        raise HTTPException(status_code=status, detail=msg)

    logger.info(f"Uploaded voice preset: {info['name']} -> {info['path']}")
    return info


@router.delete(
    "/voices/{language}/{name}",
    status_code=204,
    summary="Delete a voice preset",
    responses={
        400: {"model": ErrorResponse, "description": "Refusing to delete a path outside the voices directory."},
        404: {"model": ErrorResponse, "description": "No preset registered under that language/name."},
    },
)
async def delete_voice(
    language: str = PathParam(
        ...,
        description="ISO 639-1 language code the voice belongs to.",
        examples=["pl"],
        min_length=2,
        max_length=3,
    ),
    name: str = PathParam(
        ...,
        description="Preset name (the `<name>` part of the `<language>/<name>` key, without extension).",
        examples=["Alice"],
    ),
    voices: VoiceManager = Depends(get_voice_manager),
):
    """Delete a language-scoped voice preset by `<language>/<name>`."""
    voice_key = f"{language.lower().strip()}/{name}"
    try:
        removed = voices.delete_voice(voice_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_key}' not found")
    logger.info(f"Deleted voice preset: {voice_key}")
    return Response(status_code=204)


@router.post(
    "/voices/reload",
    response_model=ReloadVoicesResponse,
    summary="Rescan the voices directory",
    response_description="Updated preset count and sorted preset keys.",
)
async def reload_voices(voices: VoiceManager = Depends(get_voice_manager)):
    """Rescan the voices directory from disk (useful when files were added externally)."""
    voices.reload()
    return ReloadVoicesResponse(
        count=len(voices.voice_presets),
        voices=sorted(voices.voice_presets.keys()),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="VibeVoice service health",
    response_description="Model load status and device info.",
)
async def health_check(
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """Check service health and model status."""
    try:
        return HealthResponse(
            status="healthy",
            model_loaded=tts.is_loaded,
            device=tts.device if tts.device else "unknown",
            model_path=settings.vibevoice_model_path
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
            model_path=settings.vibevoice_model_path
        )
