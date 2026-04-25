"""Text chunking for long-form TTS generation.

Long generations lose quality, so we split scripts into smaller chunks at
sentence boundaries and synthesize them sequentially. Multi-speaker scripts
(``Speaker N: ...`` lines) are chunked per-turn so chunks never cross speakers
and each chunk remains a valid standalone script.

Approach ported from VibeVoice-ComfyUI's ``_split_text_into_chunks``.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_SPEAKER_LINE = re.compile(r"^\s*Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_SUB_SPLIT = re.compile(r"[,;]")


def _split_plain_text(text: str, max_words: int) -> List[str]:
    """Split a single block of text into chunks of at most ``max_words`` words."""
    text = text.strip()
    if not text:
        return []
    if max_words <= 0 or len(text.split()) <= max_words:
        return [text]

    sentences = _SENTENCE_SPLIT.split(text)
    if len(sentences) == 1:
        # Fallback: split on any period followed by space
        sentences = [s.strip() for s in text.replace(". ", ".|").split("|") if s.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_count = 0

    def flush():
        nonlocal current, current_count
        if current:
            chunks.append(" ".join(current))
            current = []
            current_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        n = len(words)

        if n > max_words:
            # Sub-split overly long sentences at commas/semicolons
            parts = [p.strip() for p in _SUB_SPLIT.split(sentence) if p.strip()]
            for part in parts:
                pn = len(part.split())
                if current_count + pn > max_words and current:
                    flush()
                current.append(part)
                current_count += pn
        else:
            if current_count + n > max_words and current:
                flush()
            current.append(sentence)
            current_count += n

    flush()
    return chunks or [text]


def _parse_script(script: str) -> List[Tuple[int, str]]:
    """Parse a multi-speaker script into ``(speaker_id, text)`` turns.

    Lines without a Speaker prefix are appended to the current turn.
    """
    turns: List[Tuple[int, List[str]]] = []
    for line in script.splitlines():
        m = _SPEAKER_LINE.match(line)
        if m:
            turns.append((int(m.group(1)), [m.group(2).strip()]))
        elif line.strip():
            if turns:
                turns[-1][1].append(line.strip())
            else:
                # No speaker label yet: assume speaker 0
                turns.append((0, [line.strip()]))
    return [(sid, " ".join(p for p in parts if p)) for sid, parts in turns]


def split_script_into_chunks(script: str, max_words: int) -> List[str]:
    """Split a (possibly multi-speaker) script into chunked sub-scripts.

    Each returned chunk is a standalone script with ``Speaker N:`` labels and
    contains at most ``max_words`` words of content per turn. If
    ``max_words <= 0`` or the script is short enough, returns ``[script]``.
    """
    script = script.strip()
    if not script:
        return []
    if max_words <= 0:
        return [script]

    has_speaker = any(_SPEAKER_LINE.match(line) for line in script.splitlines())

    if not has_speaker:
        parts = _split_plain_text(script, max_words)
        return parts or [script]

    turns = _parse_script(script)
    if not turns:
        return [script]

    chunks: List[List[str]] = [[]]
    current_count = 0

    for sid, text in turns:
        for part in _split_plain_text(text, max_words):
            pn = len(part.split())
            line = f"Speaker {sid}: {part}"
            if current_count + pn > max_words and chunks[-1]:
                chunks.append([])
                current_count = 0
            chunks[-1].append(line)
            current_count += pn

    return ["\n".join(lines) for lines in chunks if lines]
