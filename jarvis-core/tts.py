import hashlib
import io
import logging
import threading
import urllib.request
import wave
from pathlib import Path

logger = logging.getLogger("jarvis.tts")

_MODEL_DIR = Path.home() / ".jarvis" / "piper"
_ONNX_PATH = _MODEL_DIR / "jarvis.onnx"
_JSON_PATH = _MODEL_DIR / "jarvis.onnx.json"

JARVIS_ONNX_URL = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/high/jarvis-high.onnx"
JARVIS_JSON_URL = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/high/jarvis-high.onnx.json"

# Computed with: shasum -a 256 jarvis-high.onnx jarvis-high.onnx.json
JARVIS_ONNX_SHA256 = "9791877d9c099fabbf30be2825e011451c39b3431e21e81e866f5b6507e72993"
JARVIS_JSON_SHA256 = "d0b8772d81c1da2fcdfd79e90bff027f46f040450e1deb89b43a9f6b1946c5a7"

_model = None
_piper_available = None  # None = unknown, True/False after init
_init_lock = threading.Lock()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _load_model() -> bool:
    """Load Piper model, downloading if absent. Thread-safe singleton init."""
    global _model, _piper_available
    with _init_lock:
        if _piper_available is not None:
            return _piper_available
        try:
            from piper.voice import PiperVoice

            if not _ONNX_PATH.exists():
                logger.info("Downloading Piper ONNX model (~60MB)...")
                _download(JARVIS_ONNX_URL, _ONNX_PATH)
            if not _JSON_PATH.exists():
                logger.info("Downloading Piper model config...")
                _download(JARVIS_JSON_URL, _JSON_PATH)

            if _sha256(_ONNX_PATH) != JARVIS_ONNX_SHA256:
                logger.error("ONNX checksum mismatch — deleting model files for clean re-download")
                _ONNX_PATH.unlink(missing_ok=True)
                _JSON_PATH.unlink(missing_ok=True)
                _piper_available = False
                return False

            if _sha256(_JSON_PATH) != JARVIS_JSON_SHA256:
                logger.error("JSON config checksum mismatch — deleting model files for clean re-download")
                _ONNX_PATH.unlink(missing_ok=True)
                _JSON_PATH.unlink(missing_ok=True)
                _piper_available = False
                return False

            _model = PiperVoice.load(str(_ONNX_PATH), config_path=str(_JSON_PATH))
            _piper_available = True
            logger.info("Piper TTS model loaded OK")

        except ImportError:
            logger.info("piper-tts not installed — Piper TTS unavailable, using Daniel fallback")
            _piper_available = False
        except Exception as e:
            logger.warning("Piper TTS load failed: %s", e)
            _piper_available = False

    return _piper_available


def is_available() -> bool:
    """Return True only if Piper model loaded successfully."""
    return bool(_piper_available)


def synthesize(text: str) -> bytes | None:
    """Return WAV bytes for text, or None if Piper unavailable/failed."""
    if not _load_model():
        return None
    try:
        audio_io = io.BytesIO()
        with wave.open(audio_io, "wb") as wav_file:
            for chunk in _model.synthesize(text):
                if wav_file.getnframes() == 0:
                    wav_file.setnchannels(chunk.sample_channels)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setframerate(chunk.sample_rate)
                wav_file.writeframes(chunk.audio_int16_bytes)
        return audio_io.getvalue()
    except Exception as e:
        logger.warning("Piper synthesis failed: %s", e)
        return None
