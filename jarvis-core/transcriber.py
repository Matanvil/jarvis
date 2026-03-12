"""Whisper-based audio transcription for Jarvis.

Load the model once at startup with load(), then call transcribe() per audio file.
"""

import os
import re

# Whisper calls ffmpeg as a subprocess. Use the ffmpeg binary bundled with
# imageio-ffmpeg so no system ffmpeg install is required (works in DMG distribution).
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + ":" + os.environ.get("PATH", "")
except Exception:
    pass  # fall back to whatever ffmpeg is on PATH

_model = None  # module-level cache; loaded once at startup


def load() -> None:
    """Load Whisper base model into memory. Called once at startup.

    Raises ImportError if openai-whisper is not installed.
    No-op if the model is already loaded.
    """
    global _model
    if _model is not None:
        return
    import whisper
    _model = whisper.load_model("base")


def transcribe(audio_path: str) -> str:
    """Transcribe audio file at audio_path.

    Returns stripped text, or empty string if nothing was heard.
    Raises RuntimeError if model not loaded (load() not called or whisper unavailable).
    """
    if _model is None:
        raise RuntimeError("Voice transcription not available")
    result = _model.transcribe(audio_path, language="en")
    text = result["text"].strip()
    # Remove Whisper special tokens (e.g. <|nn|>, <|en|>) that occasionally leak into output
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()
    return text
