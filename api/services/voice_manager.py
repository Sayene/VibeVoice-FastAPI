"""Voice preset management and OpenAI voice mapping."""

import os
import json
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from pydub import AudioSegment
import io


def prepare_voice_sample(
    audio: np.ndarray,
    sample_rate: int = 24000,
    max_duration: float = 10.0,
    trim_silence: bool = True,
    trim_db: float = 30.0,
) -> np.ndarray:
    """Trim silence and cap length of a reference voice clip.

    Long or silence-padded reference clips cause the model to reproduce
    fragments of the reference in its output. Keeping clips short (≤ 10 s)
    and silence-free eliminates the artefact.
    """
    if trim_silence:
        try:
            audio, _ = librosa.effects.trim(audio, top_db=trim_db)
        except Exception:
            pass

    max_samples = int(max_duration * sample_rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio


class VoiceManager:
    """Manages voice presets and maps OpenAI voices to VibeVoice presets."""

    def __init__(self, voices_dir: str = "demo/voices", openai_voice_mapping: Optional[str] = None,
                 max_duration: float = 10.0, trim_silence: bool = True, trim_db: float = 30.0):
        self.voices_dir = Path(voices_dir)
        self.voice_presets: Dict[str, str] = {}
        self._max_duration = max_duration
        self._trim_silence = trim_silence
        self._trim_db = trim_db

        if openai_voice_mapping:
            try:
                self.OPENAI_VOICE_MAPPING = json.loads(openai_voice_mapping)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse OPENAI_VOICE_MAPPING: {e}. Using default mapping.")
                self.OPENAI_VOICE_MAPPING = self._get_default_mapping()
        else:
            self.OPENAI_VOICE_MAPPING = self._get_default_mapping()

        self.load_voice_presets()
    
    @staticmethod
    def _get_default_mapping() -> Dict[str, str]:
        """Get default OpenAI voice mapping using demo voices."""
        return {
            "alloy": "en-Alice_woman",
            "echo": "en-Carter_man",
            "fable": "en-Maya_woman",
            "onyx": "en-Frank_man",
            "nova": "en-Mary_woman_bgm",
            "shimmer": "en-Alice_woman"
        }
    
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

    def load_voice_presets(self):
        """Recursively scan voices directory and load available presets.

        Subdirectory names are treated as language codes (e.g. ``en/``, ``pl/``).
        Filenames may also encode language as a trailing ``_<code>`` segment
        (e.g. ``woman_1_en.mp3``). Both are fed to `_guess_language`.

        Preset name defaults to the filename stem. If two files in different
        folders share the same stem, the second one is registered under a
        folder-qualified name (e.g. ``pl/woman_1``) to avoid clobbering.
        """
        if not self.voices_dir.exists():
            print(f"Warning: Voices directory not found at {self.voices_dir}")
            return

        for file_path in sorted(self.voices_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.AUDIO_EXTENSIONS:
                continue
            # Skip hidden files (e.g. macOS .DS_Store siblings, ._resource forks)
            if any(part.startswith('.') for part in file_path.relative_to(self.voices_dir).parts):
                continue

            stem = file_path.stem
            if stem in self.voice_presets:
                # Collision: qualify with parent folder relative to voices_dir.
                rel_parent = file_path.relative_to(self.voices_dir).parent
                qualified = f"{rel_parent.as_posix()}/{stem}" if rel_parent != Path('.') else stem
                if qualified in self.voice_presets:
                    print(f"Warning: duplicate voice preset '{qualified}', skipping {file_path}")
                    continue
                self.voice_presets[qualified] = str(file_path)
            else:
                self.voice_presets[stem] = str(file_path)

        print(f"Loaded {len(self.voice_presets)} voice presets from {self.voices_dir}")
        if self.voice_presets:
            print(f"Available voices: {', '.join(sorted(self.voice_presets.keys()))}")
    
    def get_voice_path(self, voice_name: str, is_openai_voice: bool = False) -> Optional[str]:
        """
        Get path to voice preset file.
        
        Args:
            voice_name: Name of voice (OpenAI voice or VibeVoice preset name)
            is_openai_voice: Whether this is an OpenAI voice name
            
        Returns:
            Path to voice file, or None if not found
        """
        # Map OpenAI voice to VibeVoice preset if needed
        if is_openai_voice:
            voice_name = self.OPENAI_VOICE_MAPPING.get(voice_name, voice_name)
        
        return self.voice_presets.get(voice_name)
    
    def load_voice_audio(
        self,
        voice_name: str,
        is_openai_voice: bool = False,
        target_sr: int = 24000
    ) -> Optional[np.ndarray]:
        """
        Load voice audio from preset.
        
        Args:
            voice_name: Name of voice
            is_openai_voice: Whether this is an OpenAI voice name
            target_sr: Target sample rate
            
        Returns:
            Audio array, or None if voice not found
        """
        voice_path = self.get_voice_path(voice_name, is_openai_voice)
        
        if not voice_path:
            return None
        
        try:
            # Check if file format needs pydub (m4a, aac, mp3)
            file_ext = Path(voice_path).suffix.lower()
            
            if file_ext in ['.m4a', '.aac', '.mp3']:
                # Use pydub for these formats
                audio_segment = AudioSegment.from_file(voice_path)
                
                # Convert to mono if needed
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Get sample rate
                sr = audio_segment.frame_rate
                
                # Convert to numpy array (normalized to [-1, 1])
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                wav = samples / (2**15)  # Normalize 16-bit PCM to [-1, 1]
                
            else:
                # Use soundfile for wav, flac, ogg
                wav, sr = sf.read(voice_path)
                
                # Convert stereo to mono if needed
                if len(wav.shape) > 1:
                    wav = np.mean(wav, axis=1)
            
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

            wav = prepare_voice_sample(
                wav.astype(np.float32),
                sample_rate=target_sr,
                max_duration=self._max_duration,
                trim_silence=self._trim_silence,
                trim_db=self._trim_db,
            )
            return wav

        except Exception as e:
            print(f"Error loading voice {voice_name} from {voice_path}: {e}")
            return None
    
    def list_available_voices(self) -> List[Dict[str, str]]:
        """
        Get list of available voice presets.
        
        Returns:
            List of voice information dictionaries
        """
        voices = []
        for name, path in sorted(self.voice_presets.items()):
            voices.append({
                "name": name,
                "path": path,
                "language": self._guess_language(name)
            })
        return voices
    
    def list_openai_voices(self) -> List[Dict[str, str]]:
        """
        Get list of OpenAI-compatible voices.
        
        Returns:
            List of OpenAI voice information
        """
        voices = []
        for openai_name, vibevoice_preset in self.OPENAI_VOICE_MAPPING.items():
            if vibevoice_preset in self.voice_presets:
                voices.append({
                    "name": openai_name,
                    "vibevoice_preset": vibevoice_preset,
                    "available": True
                })
            else:
                voices.append({
                    "name": openai_name,
                    "vibevoice_preset": vibevoice_preset,
                    "available": False
                })
        return voices
    
    # ISO 639-1 codes → display names. Extend as needed.
    _LANG_CODES = {
        "en": "English", "zh": "Chinese", "pl": "Polish", "de": "German",
        "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
        "ru": "Russian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch",
        "cs": "Czech", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic",
        "hi": "Hindi", "in": "Indian English",
    }

    def _guess_language(self, voice_name: str) -> str:
        """Guess language from folder prefix, filename suffix, or name prefix.

        Precedence (strongest → weakest):
          1. Parent folder name (e.g. ``en/woman_1.mp3`` → en)
          2. Trailing ``_<code>`` segment in the stem (e.g. ``woman_1_en``)
          3. Legacy ``<code>-`` prefix (e.g. ``en-Alice_woman``)
        """
        # Resolve path-qualified names back to the actual on-disk path.
        path_str = self.voice_presets.get(voice_name)

        if path_str:
            try:
                rel = Path(path_str).resolve().relative_to(self.voices_dir.resolve())
                parts = rel.parts
                if len(parts) > 1:
                    first = parts[0].lower()
                    if first in self._LANG_CODES:
                        return self._LANG_CODES[first]
                stem = rel.stem
            except ValueError:
                stem = Path(path_str).stem
        else:
            stem = voice_name.rsplit('/', 1)[-1]

        # Trailing _<code>
        if '_' in stem:
            tail = stem.rsplit('_', 1)[-1].lower()
            if tail in self._LANG_CODES:
                return self._LANG_CODES[tail]

        # Legacy <code>- prefix
        if '-' in stem:
            head = stem.split('-', 1)[0].lower()
            if head in self._LANG_CODES:
                return self._LANG_CODES[head]

        return "Unknown"
    
    def add_voice_from_bytes(
        self,
        name: str,
        data: bytes,
        suffix: str,
        language: Optional[str] = None,
    ) -> Dict[str, str]:
        """Persist an uploaded voice sample to the voices directory and register it.

        If ``language`` is provided (e.g. ``"en"``), the file is stored under
        ``<voices_dir>/<language>/<name><suffix>`` — matching the convention
        used when presets are organized by language subfolder.

        Raises ValueError on invalid name/suffix or unreadable audio.
        """
        suffix = suffix.lower()
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        if suffix not in self.AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio extension '{suffix}'. Allowed: {sorted(self.AUDIO_EXTENSIONS)}"
            )

        # Sanitize: disallow path separators and hidden files; stem only.
        stem = Path(name).stem
        if not stem or stem.startswith('.') or '/' in name or '\\' in name:
            raise ValueError(f"Invalid voice name: {name!r}")

        target_dir = self.voices_dir
        if language:
            lang = language.lower().strip()
            if '/' in lang or '\\' in lang or lang.startswith('.') or not lang:
                raise ValueError(f"Invalid language: {language!r}")
            target_dir = self.voices_dir / lang

        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}{suffix}"

        # Write then validate by attempting to load it.
        target.write_bytes(data)
        try:
            self.voice_presets[stem] = str(target)
            audio = self.load_voice_audio(stem)
            if audio is None:
                raise ValueError("Uploaded file could not be decoded as audio")
        except Exception:
            # Roll back on failure.
            self.voice_presets.pop(stem, None)
            try:
                target.unlink()
            except OSError:
                pass
            raise

        return {
            "name": stem,
            "path": str(target),
            "language": self._guess_language(stem),
        }

    def delete_voice(self, name: str) -> bool:
        """Delete a voice preset from disk and unregister it. Returns True if removed."""
        path = self.voice_presets.get(name)
        if not path:
            return False

        # Safety check: must be inside voices_dir.
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.voices_dir.resolve())
        except ValueError:
            raise ValueError(f"Refusing to delete voice outside voices dir: {path}")

        try:
            resolved.unlink()
        except FileNotFoundError:
            pass
        self.voice_presets.pop(name, None)
        return True

    def reload(self) -> None:
        """Rescan the voices directory."""
        self.voice_presets.clear()
        self.load_voice_presets()

    def get_default_voice(self) -> Optional[str]:
        """Get a default voice preset name."""
        # Prefer English voices
        for name in self.voice_presets.keys():
            if name.startswith("en-"):
                return name
        
        # Return any available voice
        if self.voice_presets:
            return next(iter(self.voice_presets.keys()))
        
        return None


