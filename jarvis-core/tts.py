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

JARVIS_ONNX_URL = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/medium/jarvis-medium.onnx"
JARVIS_JSON_URL = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/medium/jarvis-medium.onnx.json"

# Computed with: shasum -a 256 jarvis.onnx jarvis.onnx.json
JARVIS_ONNX_SHA256 = "3f6534bd4050931b4c7d16ef777bafa2d90eb1e7baa8af9358623ffe609506da"
JARVIS_JSON_SHA256 = "f2c2d77f64ed6e771fc7d2defa59cd47d6bd03c3e7602c732d63ea46954f2553"

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
            from piper import PiperVoice

            if not _ONNX_PATH.exists():
                logger.info("Downloading Piper ONNX model (~60MB)...")
                _download(JARVIS_ONNX_URL, _ONNX_PATH)
            if not _JSON_PATH.exists():
                logger.info("Downloading Piper model config...")
                _download(JARVIS_JSON_URL, _JSON_PATH)

            if _sha256(_ONNX_PATH) != JARVIS_ONNX_SHA256:
                logger.error("ONNX checksum mismatch — deleting corrupted file")
                _ONNX_PATH.unlink(missing_ok=True)
                _piper_available = False
                return False

            if _sha256(_JSON_PATH) != JARVIS_JSON_SHA256:
                logger.error("JSON config checksum mismatch — deleting corrupted file")
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
            _model.synthesize(text, wav_file)
        return audio_io.getvalue()
    except Exception as e:
        logger.warning("Piper synthesis failed: %s", e)
        return None
