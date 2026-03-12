"""Whisper-based audio transcription for Jarvis.

Load the model once at startup with load(), then call transcribe() per audio file.
"""

import os
import re
import subprocess

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


def _load_audio(audio_path: str) -> "np.ndarray":
    """Decode audio file to a 16kHz mono float32 numpy array.

    Uses imageio-ffmpeg's bundled binary (full path) so no system ffmpeg is needed.
    Falls back to passing the path directly to Whisper if imageio-ffmpeg is unavailable.
    """
    import numpy as np
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return audio_path  # let Whisper handle it with whatever ffmpeg is on PATH
    cmd = [
        ffmpeg_exe, "-nostdin", "-threads", "0",
        "-i", audio_path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe(audio_path: str) -> str:
    """Transcribe audio file at audio_path.

    Returns stripped text, or empty string if nothing was heard.
    Raises RuntimeError if model not loaded (load() not called or whisper unavailable).
    """
    if _model is None:
        raise RuntimeError("Voice transcription not available")
    audio = _load_audio(audio_path)
    result = _model.transcribe(audio, language="en", fp16=False)
    text = result["text"].strip()
    # Remove Whisper special tokens (e.g. <|nn|>, <|en|>) that occasionally leak into output
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()
    return text
