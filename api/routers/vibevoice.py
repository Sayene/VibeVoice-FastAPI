"""VibeVoice-specific TTS endpoints with multi-speaker support."""

import base64
import logging
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import Response

from api.models import (
    VibeVoiceGenerateRequest,
    VibeVoiceGenerateResponse,
    VoiceListResponse,
    HealthResponse
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


@router.post("/generate")
async def generate_speech(
    request: VibeVoiceGenerateRequest,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Generate multi-speaker speech with VibeVoice-specific features.
    
    Supports:
    - Multi-speaker dialogue (up to 4 speakers)
    - Custom voice samples via base64 or presets
    - CFG scale control
    - Inference step control
    - Real-time streaming via SSE
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
                    
                    # Resample if needed
                    if sr != 24000:
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)
                    
                    # Convert to mono if needed
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
                # Load from preset
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
        
        # Extract voice presets for logging
        voice_list = []
        for speaker_config in sorted(request.speakers, key=lambda s: s.speaker_id):
            if speaker_config.voice_preset:
                voice_list.append(f"speaker{speaker_config.speaker_id}={speaker_config.voice_preset}")
            else:
                voice_list.append(f"speaker{speaker_config.speaker_id}=base64_audio")
        voices_str = ", ".join(voice_list)
        
        # Get actual inference_steps value (request value or default from settings)
        actual_inference_steps = request.inference_steps if request.inference_steps is not None else settings.vibevoice_inference_steps
        
        # Generate speech
        if request.stream:
            # Return streaming response
            # Note: For streaming, we log before generation starts since timing is async
            text_preview = request.script[:100] + "..." if len(request.script) > 100 else request.script
            logger.info(
                f"Generating speech (streaming) - Text: {text_preview} | Voices: {voices_str} | "
                f"Model: {settings.vibevoice_model_path} | CFG: {request.cfg_scale} | "
                f"Steps: {actual_inference_steps} | Seed: {request.seed if request.seed is not None else 'None'}"
            )
            
            audio_stream = tts.generate_speech(
                text=request.script,
                voice_samples=voice_samples,
                cfg_scale=request.cfg_scale,
                inference_steps=request.inference_steps,
                seed=request.seed,
                stream=True
            )
            
            # For streaming, we can't measure exact time, but log start
            return create_streaming_response(
                audio_stream,
                format=request.response_format,
                sample_rate=24000,
                use_sse=True
            )
        
        else:
            # Generate all at once
            start_time = time.time()
            audio = tts.generate_speech(
                text=request.script,
                voice_samples=voice_samples,
                cfg_scale=request.cfg_scale,
                inference_steps=request.inference_steps,
                seed=request.seed,
                stream=False
            )
            generation_time = time.time() - start_time
            
            # Calculate audio duration
            audio_duration = get_audio_duration(audio, sample_rate=24000)
            
            # Log generation details at INFO level
            text_preview = request.script[:100] + "..." if len(request.script) > 100 else request.script
            logger.info(
                f"Generated speech - Text: {text_preview} | Voices: {voices_str} | "
                f"Model: {settings.vibevoice_model_path} | CFG: {request.cfg_scale} | "
                f"Steps: {actual_inference_steps} | Seed: {request.seed if request.seed is not None else 'None'} | "
                f"Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
            )
            
            # Convert to requested format
            audio_bytes = audio_to_bytes(
                audio,
                sample_rate=24000,
                format=request.response_format
            )
            
            # Use the already calculated duration
            duration = audio_duration
            
            # Return audio response
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


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(voices: VoiceManager = Depends(get_voice_manager)):
    """
    List all available VibeVoice presets.
    """
    try:
        available_voices = voices.list_available_voices()
        return VoiceListResponse(voices=available_voices)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voices", status_code=201)
async def upload_voice(
    file: UploadFile = File(..., description="Audio file (wav/mp3/flac/ogg/m4a/aac)"),
    name: str | None = Form(None, description="Voice preset name (defaults to uploaded filename stem)"),
    language: str | None = Form(
        None,
        description="Optional language code (e.g. 'en', 'pl'). If set, the file is stored under <voices_dir>/<language>/.",
    ),
    voices: VoiceManager = Depends(get_voice_manager),
):
    """Upload a new voice preset. Persists to the voices directory.

    When ``language`` is provided, the file lands in a language subfolder so
    it follows the same organization as on-disk presets.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix or ""
    preset_name = name or Path(filename).stem
    if not preset_name:
        raise HTTPException(status_code=400, detail="Could not determine voice name")

    if preset_name in voices.voice_presets:
        raise HTTPException(
            status_code=409,
            detail=f"Voice '{preset_name}' already exists. Delete it first or choose a different name.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        info = voices.add_voice_from_bytes(preset_name, data, suffix, language=language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Uploaded voice preset: {info['name']} -> {info['path']}")
    return info


@router.delete("/voices/{voice_name}", status_code=204)
async def delete_voice(
    voice_name: str,
    voices: VoiceManager = Depends(get_voice_manager),
):
    """Delete a voice preset by name."""
    try:
        removed = voices.delete_voice(voice_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    logger.info(f"Deleted voice preset: {voice_name}")
    return Response(status_code=204)


@router.post("/voices/reload")
async def reload_voices(voices: VoiceManager = Depends(get_voice_manager)):
    """Rescan the voices directory from disk (useful if files were added externally)."""
    voices.reload()
    return {"count": len(voices.voice_presets), "voices": sorted(voices.voice_presets.keys())}


@router.get("/health", response_model=HealthResponse)
async def health_check(
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Check service health and model status.
    """
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


