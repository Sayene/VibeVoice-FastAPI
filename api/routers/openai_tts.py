"""OpenAI-compatible TTS endpoint."""

import logging
import time
from fastapi import APIRouter, HTTPException, Depends
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


@router.post("/speech")
async def create_speech(
    request: OpenAITTSRequest,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Generate speech from text using OpenAI-compatible API.
    
    This endpoint mimics the OpenAI TTS API for compatibility with existing clients.
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
        audio = tts.generate_speech(
            text=formatted_script,
            voice_samples=[voice_audio],
            cfg_scale=settings.default_cfg_scale,
            stream=False  # For OpenAI compatibility, generate all at once
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
            f"CFG: {settings.default_cfg_scale} | Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
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


@router.get("/voices")
async def list_voices(
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    List all available voices in OpenAI-compatible format.
    
    Returns OpenAI standard voices (if their mapped presets exist) plus all custom voices.
    """
    try:
        voice_list = []
        
        # Add OpenAI standard voices (if their mapped presets exist)
        for openai_name, vibevoice_preset in voices.OPENAI_VOICE_MAPPING.items():
            if vibevoice_preset in voices.voice_presets:
                voice_list.append({
                    "id": openai_name,
                    "object": "voice",
                    "name": openai_name
                })
        
        # Add all custom voices from VOICES_DIR
        all_voices = voices.list_available_voices()
        for voice in all_voices:
            # Skip if already added as OpenAI voice
            if voice["name"] not in voices.OPENAI_VOICE_MAPPING.values():
                voice_list.append({
                    "id": voice["name"],
                    "object": "voice",
                    "name": voice["name"]
                })
        
        return {
            "object": "list",
            "data": voice_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

