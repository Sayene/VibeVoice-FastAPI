"""FastAPI application for VibeVoice TTS API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.routers import openai_tts, vibevoice


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.normalized_log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting VibeVoice API server...")
    logger.info(f"Model path: {settings.vibevoice_model_path}")
    logger.info(f"Device: {settings.vibevoice_device}")
    logger.info(f"Voices directory: {settings.voices_dir}")
    
    # Initialize voice manager
    logger.info("Initializing voice manager...")
    voice_manager = VoiceManager(
        voices_dir=settings.voices_dir,
        openai_voice_mapping=settings.openai_voice_mapping,
        max_duration=settings.voice_sample_max_duration,
        trim_silence=settings.voice_sample_trim_silence,
        trim_db=settings.voice_sample_trim_db,
    )
    
    # Initialize TTS service
    logger.info("Initializing TTS service...")
    tts_service = TTSService(settings)
    
    # Load model
    logger.info("Loading VibeVoice model (this may take a few minutes)...")
    try:
        tts_service.load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Set global service instances in routers
    openai_tts.tts_service = tts_service
    openai_tts.voice_manager = voice_manager
    vibevoice.tts_service = tts_service
    vibevoice.voice_manager = voice_manager
    
    logger.info("API server ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VibeVoice API server...")


# Create FastAPI app
app = FastAPI(
    title="VibeVoice TTS API",
    description="OpenAI-compatible Text-to-Speech API powered by VibeVoice",
    version="0.5.6",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(openai_tts.router)
app.include_router(vibevoice.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VibeVoice TTS API",
        "version": "0.5.6",
        "description": "OpenAI-compatible Text-to-Speech API powered by VibeVoice",
        "endpoints": {
            "openai_compatible": {
                "speech": "/v1/audio/speech",
                "voices": "/v1/audio/voices"
            },
            "vibevoice_extended": {
                "generate": "/v1/vibevoice/generate",
                "list_voices": "GET /v1/vibevoice/voices?language=<iso639-1>",
                "upload_voice": "POST /v1/vibevoice/voices (multipart: language, file, name)",
                "delete_voice": "DELETE /v1/vibevoice/voices/{language}/{name}",
                "reload_voices": "POST /v1/vibevoice/voices/reload",
                "health": "/v1/vibevoice/health"
            },
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy"}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": type(exc).__name__,
                "detail": str(exc)
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.normalized_log_level.lower()
    )

