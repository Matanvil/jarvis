import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tts


@pytest.fixture(autouse=True)
def reset_tts():
    """Reset module-level singleton between tests."""
    tts._piper_bin = None
    tts._piper_available = None
    yield
    tts._piper_bin = None
    tts._piper_available = None


def _fake_subprocess_run(fake_wav: bytes):
    """Return a subprocess.run mock that writes fake_wav to --output_file."""
    def _run(cmd, **kwargs):
        idx = cmd.index("--output_file")
        Path(cmd[idx + 1]).write_bytes(fake_wav)
        return MagicMock(returncode=0)
    return _run


_FAKE_WAV = b"RIFF" + b"\x00" * 40


def test_synthesize_returns_bytes_when_piper_available(tmp_path):
    bin_path = tmp_path / "piper"
    bin_path.touch()
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"fake")

    tts._piper_available = True
    tts._piper_bin = bin_path

    with patch("tts._ONNX_PATH", onnx), \
         patch("subprocess.run", side_effect=_fake_subprocess_run(_FAKE_WAV)):
        result = tts.synthesize("Hello JARVIS")

    assert result == _FAKE_WAV


def test_synthesize_returns_none_when_piper_unavailable():
    tts._piper_available = False
    result = tts.synthesize("Hello")
    assert result is None


def test_synthesize_returns_none_on_subprocess_error(tmp_path):
    bin_path = tmp_path / "piper"
    bin_path.touch()
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"fake")

    tts._piper_available = True
    tts._piper_bin = bin_path

    mock_result = MagicMock(returncode=1, stderr=b"synthesis error")
    with patch("tts._ONNX_PATH", onnx), \
         patch("subprocess.run", return_value=mock_result):
        result = tts.synthesize("Hello")

    assert result is None


def test_synthesize_returns_none_on_subprocess_exception(tmp_path):
    bin_path = tmp_path / "piper"
    bin_path.touch()
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"fake")

    tts._piper_available = True
    tts._piper_bin = bin_path

    with patch("tts._ONNX_PATH", onnx), \
         patch("subprocess.run", side_effect=RuntimeError("crash")):
        result = tts.synthesize("Hello")

    assert result is None


def test_synthesize_returns_none_on_checksum_mismatch(tmp_path):
    bin_path = tmp_path / "piper"
    bin_path.touch()
    bin_path.chmod(0o755)
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"corrupted")
    json_f = tmp_path / "jarvis.onnx.json"
    json_f.write_text("{}")

    with patch("tts._ensure_piper_binary", return_value=bin_path), \
         patch("tts._ONNX_PATH", onnx), \
         patch("tts._JSON_PATH", json_f), \
         patch("tts._sha256", return_value="badhash"), \
         patch("tts.JARVIS_ONNX_SHA256", "correcthash"), \
         patch("tts.JARVIS_JSON_SHA256", "correcthash"):
        result = tts.synthesize("Hello")

    assert result is None
    assert not onnx.exists()


def test_load_model_thread_safe(tmp_path):
    """Concurrent calls result in exactly one binary + model init."""
    call_count = {"n": 0}
    bin_path = tmp_path / "piper"
    bin_path.touch()
    bin_path.chmod(0o755)
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"fake")
    json_f = tmp_path / "jarvis.onnx.json"
    json_f.write_text("{}")

    def slow_ensure_binary():
        call_count["n"] += 1
        import time
        time.sleep(0.05)
        return bin_path

    results = []

    def worker():
        results.append(tts._load_model())

    with patch("tts._ensure_piper_binary", side_effect=slow_ensure_binary), \
         patch("tts._ONNX_PATH", onnx), \
         patch("tts._JSON_PATH", json_f), \
         patch("tts._sha256", return_value="deadbeef"), \
         patch("tts.JARVIS_ONNX_SHA256", "deadbeef"), \
         patch("tts.JARVIS_JSON_SHA256", "deadbeef"):
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert call_count["n"] == 1
    assert all(r is True for r in results)


def test_piper_available_false_on_binary_unavailable():
    """_piper_available is False when no binary for this platform."""
    with patch("tts._ensure_piper_binary", return_value=None):
        tts._load_model()
    assert tts.is_available() is False


def test_piper_available_false_on_download_failure(tmp_path):
    """_piper_available is False when model download fails."""
    bin_path = tmp_path / "piper"
    bin_path.touch()
    bin_path.chmod(0o755)
    missing = tmp_path / "jarvis.onnx"

    with patch("tts._ensure_piper_binary", return_value=bin_path), \
         patch("tts._ONNX_PATH", missing), \
         patch("tts._JSON_PATH", tmp_path / "jarvis.onnx.json"), \
         patch("tts._download", side_effect=OSError("download failed")):
        tts._load_model()
    assert tts.is_available() is False


def test_piper_available_false_on_checksum_mismatch(tmp_path):
    """_piper_available is False when SHA-256 does not match."""
    bin_path = tmp_path / "piper"
    bin_path.touch()
    bin_path.chmod(0o755)
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"corrupted")
    json_f = tmp_path / "jarvis.onnx.json"
    json_f.write_text("{}")

    with patch("tts._ensure_piper_binary", return_value=bin_path), \
         patch("tts._ONNX_PATH", onnx), \
         patch("tts._JSON_PATH", json_f), \
         patch("tts._sha256", return_value="badhash"), \
         patch("tts.JARVIS_ONNX_SHA256", "goodhash"), \
         patch("tts.JARVIS_JSON_SHA256", "goodhash"):
        tts._load_model()
    assert tts.is_available() is False


def test_piper_available_false_on_load_exception():
    """_piper_available is False when _ensure_piper_binary raises."""
    with patch("tts._ensure_piper_binary", side_effect=RuntimeError("unexpected")):
        tts._load_model()
    assert tts.is_available() is False


def test_is_available_false_before_init():
    assert tts.is_available() is False
