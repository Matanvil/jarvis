import hashlib
import logging
import os
import platform
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

logger = logging.getLogger("jarvis.tts")

_MODEL_DIR = Path.home() / ".jarvis" / "piper"
_ONNX_PATH = _MODEL_DIR / "jarvis.onnx"
_JSON_PATH = _MODEL_DIR / "jarvis.onnx.json"
_PIPER_DIR = _MODEL_DIR / "bin"

JARVIS_ONNX_URL = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/medium/jarvis-medium.onnx"
JARVIS_JSON_URL = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/medium/jarvis-medium.onnx.json"

# Computed with: shasum -a 256 jarvis.onnx jarvis.onnx.json
JARVIS_ONNX_SHA256 = "3f6534bd4050931b4c7d16ef777bafa2d90eb1e7baa8af9358623ffe609506da"
JARVIS_JSON_SHA256 = "f2c2d77f64ed6e771fc7d2defa59cd47d6bd03c3e7602c732d63ea46954f2553"

_PIPER_BINARY_URLS = {
    ("Darwin", "arm64"): "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_aarch64.tar.gz",
    ("Darwin", "x86_64"): "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_x86_64.tar.gz",
    ("Linux", "aarch64"): "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz",
    ("Linux", "x86_64"): "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz",
}

_piper_bin: Path | None = None
_piper_available = None  # None = unknown, True/False after init
_init_lock = threading.Lock()


def _get_binary_url() -> str | None:
    system = platform.system()
    machine = platform.machine()
    return _PIPER_BINARY_URLS.get((system, machine))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _find_piper_binary(search_dir: Path) -> Path | None:
    """Find the piper executable in an extracted directory."""
    for candidate in search_dir.rglob("piper"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _ensure_piper_binary() -> Path | None:
    """Return path to piper binary, downloading and extracting if needed."""
    if _PIPER_DIR.exists():
        existing = _find_piper_binary(_PIPER_DIR)
        if existing:
            return existing

    url = _get_binary_url()
    if url is None:
        logger.warning(
            "No piper binary for this platform (%s %s)",
            platform.system(),
            platform.machine(),
        )
        return None

    archive = _MODEL_DIR / "piper.tar.gz"
    logger.info("Downloading piper binary...")
    _download(url, archive)

    _PIPER_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(_PIPER_DIR)
    archive.unlink(missing_ok=True)

    return _find_piper_binary(_PIPER_DIR)


def _load_model() -> bool:
    """Ensure piper binary + model files are ready. Thread-safe singleton init."""
    global _piper_bin, _piper_available
    with _init_lock:
        if _piper_available is not None:
            return _piper_available
        try:
            bin_path = _ensure_piper_binary()
            if bin_path is None:
                logger.info("piper binary unavailable — falling back to Daniel")
                _piper_available = False
                return False

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

            _piper_bin = bin_path
            _piper_available = True
            logger.info("Piper TTS ready (binary: %s)", bin_path)

        except Exception as e:
            logger.warning("Piper TTS load failed: %s", e)
            _piper_available = False

    return _piper_available


def is_available() -> bool:
    """Return True only if Piper is ready."""
    return bool(_piper_available)


def synthesize(text: str) -> bytes | None:
    """Return WAV bytes for text, or None if Piper unavailable/failed."""
    if not _load_model():
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = Path(f.name)
        try:
            env = os.environ.copy()
            espeak_data = _piper_bin.parent / "espeak-ng-data"
            if espeak_data.exists():
                env["ESPEAK_DATA_PATH"] = str(espeak_data)
            result = subprocess.run(
                [str(_piper_bin), "--model", str(_ONNX_PATH), "--output_file", str(out_path)],
                input=text.encode(),
                capture_output=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0:
                logger.warning(
                    "Piper synthesis failed (exit %d): %s",
                    result.returncode,
                    result.stderr.decode(errors="replace"),
                )
                return None
            return out_path.read_bytes()
        finally:
            out_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Piper synthesis failed: %s", e)
        return None
