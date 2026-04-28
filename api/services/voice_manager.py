"""Voice preset management and OpenAI voice mapping.

Voice presets are organized on disk by ISO 639-1 language code:

    <voices_dir>/<lang>/<name>.<ext>      e.g.  voices/pl/Alice.wav

The preset key (used everywhere in the API) is ``<lang>/<name>``. Files
sitting directly under ``<voices_dir>`` (no language folder) are still
loaded for backward compatibility and keyed by their bare stem.
"""

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
    max_duration: float = 0.0,
    trim_silence: bool = False,
    trim_db: float = 30.0,
) -> np.ndarray:
    """Optional reference-voice preprocessing.

    Disabled by default to match the ComfyUI VibeVoice front-end and the
    Microsoft reference inference demo, both of which feed the raw resampled
    mono clip straight into the processor (the processor's own
    ``AudioNormalizer`` handles RMS normalisation to -25 dBFS).

    - ``trim_silence`` (off by default): strip leading/trailing silence with
      ``librosa.effects.trim(top_db=trim_db)``. Can shave off soft phoneme
      boundaries and degrade pronunciation/accent fidelity on non-English
      voices, so leave off unless you know your clips have long silences.
    - ``max_duration`` (0 = no cap): hard-cap clip length in seconds. Cutting
      a reference shorter throws away phonemes the model uses to generalise
      the voice; only useful if your clips are unusually long.
    """
    if trim_silence:
        try:
            audio, _ = librosa.effects.trim(audio, top_db=trim_db)
        except Exception:
            pass

    if max_duration and max_duration > 0:
        max_samples = int(max_duration * sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    return audio


class VoiceManager:
    """Manages voice presets and maps OpenAI voices to VibeVoice presets."""

    # ISO 639-1 codes → display names. Extend as needed.
    _LANG_CODES = {
        "en": "English", "zh": "Chinese", "pl": "Polish", "de": "German",
        "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
        "ru": "Russian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch",
        "cs": "Czech", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic",
        "hi": "Hindi", "in": "Indian English",
    }

    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

    def __init__(self, voices_dir: str = "demo/voices", openai_voice_mapping: Optional[str] = None,
                 max_duration: float = 0.0, trim_silence: bool = False, trim_db: float = 30.0):
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
        """Default OpenAI → preset mapping (legacy bare-stem demo voices)."""
        return {
            "alloy": "en-Alice_woman",
            "echo": "en-Carter_man",
            "fable": "en-Maya_woman",
            "onyx": "en-Frank_man",
            "nova": "en-Mary_woman_bgm",
            "shimmer": "en-Alice_woman",
        }

    @classmethod
    def _validate_lang_code(cls, language: str) -> str:
        """Normalize and validate a language code. Returns the lowercased code."""
        if not language:
            raise ValueError("Language code is required")
        lang = language.lower().strip()
        if not lang or '/' in lang or '\\' in lang or lang.startswith('.'):
            raise ValueError(f"Invalid language code: {language!r}")
        if lang not in cls._LANG_CODES:
            allowed = ", ".join(sorted(cls._LANG_CODES))
            raise ValueError(
                f"Unknown language code {language!r}. Supported: {allowed}"
            )
        return lang

    def load_voice_presets(self):
        """Recursively scan ``voices_dir`` and register every audio file.

        Files inside a language folder are keyed as ``<lang>/<stem>``. Files
        directly under ``voices_dir`` are keyed by bare stem (legacy demo
        voices). Hidden files and unsupported extensions are skipped.
        """
        if not self.voices_dir.exists():
            print(f"Warning: Voices directory not found at {self.voices_dir}")
            return

        for file_path in sorted(self.voices_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.AUDIO_EXTENSIONS:
                continue
            rel = file_path.relative_to(self.voices_dir)
            if any(part.startswith('.') for part in rel.parts):
                continue

            parts = rel.parts
            if len(parts) >= 2 and parts[0].lower() in self._LANG_CODES:
                key = f"{parts[0].lower()}/{file_path.stem}"
            else:
                key = file_path.stem

            if key in self.voice_presets:
                print(f"Warning: duplicate voice preset '{key}', skipping {file_path}")
                continue
            self.voice_presets[key] = str(file_path)

        print(f"Loaded {len(self.voice_presets)} voice presets from {self.voices_dir}")
        if self.voice_presets:
            print(f"Available voices: {', '.join(sorted(self.voice_presets.keys()))}")

    def get_voice_path(self, voice_name: str, is_openai_voice: bool = False) -> Optional[str]:
        """Resolve a preset key (or OpenAI voice name) to an on-disk path."""
        if is_openai_voice:
            voice_name = self.OPENAI_VOICE_MAPPING.get(voice_name, voice_name)
        return self.voice_presets.get(voice_name)

    def load_voice_audio(
        self,
        voice_name: str,
        is_openai_voice: bool = False,
        target_sr: int = 24000,
    ) -> Optional[np.ndarray]:
        """Decode the preset's audio into a float32 mono ndarray at ``target_sr``."""
        voice_path = self.get_voice_path(voice_name, is_openai_voice)
        if not voice_path:
            return None

        try:
            file_ext = Path(voice_path).suffix.lower()

            if file_ext in ['.m4a', '.aac', '.mp3']:
                audio_segment = AudioSegment.from_file(voice_path)
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                sr = audio_segment.frame_rate
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                wav = samples / (2**15)
            else:
                wav, sr = sf.read(voice_path)
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

    def resolve_voice_for_language(
        self,
        voice_name: str,
        language: str,
        is_openai_voice: bool = False,
    ) -> Optional[str]:
        """Find a preset key under ``<voices_dir>/<language>/`` matching ``voice_name``.

        Match order: exact ``<lang>/<voice_name>`` key, then for OpenAI voices
        the mapped preset (with any legacy ``<code>-`` prefix stripped).
        """
        try:
            lang = self._validate_lang_code(language)
        except ValueError:
            return None

        candidates = [voice_name]
        if is_openai_voice:
            mapped = self.OPENAI_VOICE_MAPPING.get(voice_name)
            if mapped:
                candidates.append(mapped)
                # Mapped value may itself be "<lang>/<stem>" or legacy "<code>-<stem>".
                if "/" in mapped:
                    candidates.append(mapped.split("/", 1)[1])
                elif "-" in mapped:
                    head, rest = mapped.split("-", 1)
                    if head.lower() in self._LANG_CODES:
                        candidates.append(rest)

        for cand in candidates:
            stem = cand.rsplit("/", 1)[-1]
            key = f"{lang}/{stem}"
            if key in self.voice_presets:
                return key
        return None

    def list_available_voices(self, language: Optional[str] = None) -> List[Dict[str, str]]:
        """Return registered presets, optionally filtered by language code.

        ``language`` is an ISO 639-1 code (e.g. ``"pl"``). When supplied, only
        presets stored under that language folder are returned.
        """
        lang_filter: Optional[str] = None
        if language:
            lang_filter = self._validate_lang_code(language)

        voices = []
        for name, path in sorted(self.voice_presets.items()):
            voice_lang_code = self._language_code_for(name)
            if lang_filter and voice_lang_code != lang_filter:
                continue
            voices.append({
                "name": name,
                "path": path,
                "language": self._LANG_CODES.get(voice_lang_code, "Unknown") if voice_lang_code else "Unknown",
                "language_code": voice_lang_code or "",
            })
        return voices

    def list_openai_voices(self) -> List[Dict[str, str]]:
        """Return the OpenAI voice mapping with availability flags."""
        voices = []
        for openai_name, vibevoice_preset in self.OPENAI_VOICE_MAPPING.items():
            voices.append({
                "name": openai_name,
                "vibevoice_preset": vibevoice_preset,
                "available": vibevoice_preset in self.voice_presets,
            })
        return voices

    def _language_code_for(self, voice_key: str) -> Optional[str]:
        """Return the ISO 639-1 code of a registered preset, or None."""
        # Composite key: "<lang>/<stem>"
        if "/" in voice_key:
            head = voice_key.split("/", 1)[0].lower()
            if head in self._LANG_CODES:
                return head

        # Legacy bare-stem keys: try filename hints.
        stem = voice_key
        if '_' in stem:
            tail = stem.rsplit('_', 1)[-1].lower()
            if tail in self._LANG_CODES:
                return tail
        if '-' in stem:
            head = stem.split('-', 1)[0].lower()
            if head in self._LANG_CODES:
                return head
        return None

    def _guess_language(self, voice_name: str) -> str:
        """Display name (e.g. ``"Polish"``) for a preset key."""
        code = self._language_code_for(voice_name)
        return self._LANG_CODES.get(code, "Unknown") if code else "Unknown"

    def add_voice_from_bytes(
        self,
        name: str,
        data: bytes,
        suffix: str,
        language: str,
    ) -> Dict[str, str]:
        """Persist an uploaded voice sample under ``<voices_dir>/<language>/``.

        Registers the file as ``<language>/<stem>``. ``language`` is required
        and must be a known ISO 639-1 code.
        Raises ValueError on invalid name/suffix/language or unreadable audio.
        """
        suffix = suffix.lower()
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        if suffix not in self.AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio extension '{suffix}'. Allowed: {sorted(self.AUDIO_EXTENSIONS)}"
            )

        stem = Path(name).stem
        if not stem or stem.startswith('.') or '/' in name or '\\' in name:
            raise ValueError(f"Invalid voice name: {name!r}")

        lang = self._validate_lang_code(language)
        target_dir = self.voices_dir / lang
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}{suffix}"
        key = f"{lang}/{stem}"

        if key in self.voice_presets or target.exists():
            raise ValueError(
                f"Voice '{key}' already exists. Delete it first or choose a different name."
            )

        target.write_bytes(data)
        try:
            self.voice_presets[key] = str(target)
            audio = self.load_voice_audio(key)
            if audio is None:
                raise ValueError("Uploaded file could not be decoded as audio")
        except Exception:
            self.voice_presets.pop(key, None)
            try:
                target.unlink()
            except OSError:
                pass
            raise

        return {
            "name": key,
            "path": str(target),
            "language": self._LANG_CODES.get(lang, "Unknown"),
            "language_code": lang,
        }

    def delete_voice(self, name: str) -> bool:
        """Delete a preset from disk and unregister it. Returns True if removed."""
        path = self.voice_presets.get(name)
        if not path:
            return False

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
        """Pick a sensible default preset (prefers English)."""
        for name in self.voice_presets:
            if name.startswith("en/") or name.startswith("en-"):
                return name
        if self.voice_presets:
            return next(iter(self.voice_presets))
        return None
