"""Whisper-based audio transcription for Jarvis.

Load the model once at startup with load(), then call transcribe() per audio file.
"""

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
    result = _model.transcribe(audio_path)
    return result["text"].strip()
