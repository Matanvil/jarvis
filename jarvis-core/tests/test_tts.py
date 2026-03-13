import io
import threading
import wave
import pytest
from unittest.mock import MagicMock, patch
import tts


@pytest.fixture(autouse=True)
def reset_tts():
    """Reset module-level singleton between tests."""
    tts._model = None
    tts._piper_available = None
    yield
    tts._model = None
    tts._piper_available = None


def _make_mock_voice():
    """Return a mock PiperVoice that writes a minimal valid WAV."""
    mock_voice = MagicMock()

    def fake_synthesize(text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00" * 200)

    mock_voice.synthesize.side_effect = fake_synthesize
    return mock_voice


def _mock_piper_module(voice):
    mock_piper = MagicMock()
    mock_piper.PiperVoice.load.return_value = voice
    return mock_piper


def test_synthesize_returns_bytes_when_piper_available(tmp_path):
    mock_voice = _make_mock_voice()
    mock_piper = _mock_piper_module(mock_voice)
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"fake")
    json_f = tmp_path / "jarvis.onnx.json"
    json_f.write_text("{}")

    with patch.dict("sys.modules", {"piper": mock_piper}), \
         patch("tts._ONNX_PATH", onnx), \
         patch("tts._JSON_PATH", json_f), \
         patch("tts._sha256", return_value="deadbeef"), \
         patch("tts.JARVIS_ONNX_SHA256", "deadbeef"), \
         patch("tts.JARVIS_JSON_SHA256", "deadbeef"):
        result = tts.synthesize("Hello JARVIS")

    assert isinstance(result, bytes)
    assert len(result) > 44  # WAV header is 44 bytes


def test_synthesize_returns_none_when_piper_not_installed():
    import sys
    # Make 'from piper import PiperVoice' raise ImportError
    with patch.dict("sys.modules", {"piper": None}):
        result = tts.synthesize("Hello")

    assert result is None


def test_synthesize_returns_none_on_synthesis_exception(tmp_path):
    mock_voice = MagicMock()
    mock_voice.synthesize.side_effect = RuntimeError("synthesis failed")
    mock_piper = _mock_piper_module(mock_voice)
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"fake")
    json_f = tmp_path / "jarvis.onnx.json"
    json_f.write_text("{}")

    with patch.dict("sys.modules", {"piper": mock_piper}), \
         patch("tts._ONNX_PATH", onnx), \
         patch("tts._JSON_PATH", json_f), \
         patch("tts._sha256", return_value="deadbeef"), \
         patch("tts.JARVIS_ONNX_SHA256", "deadbeef"), \
         patch("tts.JARVIS_JSON_SHA256", "deadbeef"):
        result = tts.synthesize("Hello")

    assert result is None


def test_synthesize_returns_none_on_checksum_mismatch(tmp_path):
    mock_piper = MagicMock()
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"corrupted")
    json_f = tmp_path / "jarvis.onnx.json"
    json_f.write_text("{}")

    with patch.dict("sys.modules", {"piper": mock_piper}), \
         patch("tts._ONNX_PATH", onnx), \
         patch("tts._JSON_PATH", json_f), \
         patch("tts._sha256", return_value="badhash"), \
         patch("tts.JARVIS_ONNX_SHA256", "correcthash"), \
         patch("tts.JARVIS_JSON_SHA256", "correcthash"):
        result = tts.synthesize("Hello")

    assert result is None
    # File should be deleted after checksum failure
    assert not onnx.exists()


def test_load_model_thread_safe(tmp_path):
    """Concurrent calls result in exactly one initialization.

    Patches are applied OUTSIDE the threads so all threads share the same
    patched state. Per-thread patch contexts would race with each other and
    undermine the single-init assertion.
    """
    call_count = {"n": 0}
    mock_voice = _make_mock_voice()

    def slow_load(*args, **kwargs):
        call_count["n"] += 1
        import time
        time.sleep(0.05)
        return mock_voice

    mock_piper = MagicMock()
    mock_piper.PiperVoice.load.side_effect = slow_load
    onnx = tmp_path / "jarvis.onnx"
    onnx.write_bytes(b"fake")
    json_f = tmp_path / "jarvis.onnx.json"
    json_f.write_text("{}")

    results = []

    def worker():
        results.append(tts._load_model())

    with patch.dict("sys.modules", {"piper": mock_piper}), \
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


def test_piper_available_false_on_all_failure_paths():
    """_piper_available is False after import error, not None."""
    with patch.dict("sys.modules", {"piper": None}):
        tts._load_model()
    assert tts.is_available() is False


def test_is_available_false_before_init():
    assert tts.is_available() is False
