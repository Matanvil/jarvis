import sys
import pytest
from unittest.mock import MagicMock, patch


def setup_function():
    """Reset _model before each test to avoid cross-test state."""
    import transcriber
    transcriber._model = None


def test_load_calls_whisper_load_model():
    import transcriber
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        transcriber.load()

    mock_whisper.load_model.assert_called_once_with("base")
    assert transcriber._model is mock_model


def test_transcribe_returns_stripped_text():
    import transcriber
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "  run the tests  "}
    transcriber._model = mock_model
    fake_audio = object()

    with patch("transcriber._load_audio", return_value=fake_audio):
        result = transcriber.transcribe("/tmp/audio.ogg")

    assert result == "run the tests"
    mock_model.transcribe.assert_called_once_with(fake_audio, language="en", fp16=False)


def test_transcribe_returns_empty_string_for_whitespace():
    import transcriber
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "   "}
    transcriber._model = mock_model

    with patch("transcriber._load_audio", return_value=object()):
        result = transcriber.transcribe("/tmp/audio.ogg")

    assert result == ""


def test_transcribe_raises_runtime_error_when_model_not_loaded():
    import transcriber
    transcriber._model = None

    with pytest.raises(RuntimeError, match="Voice transcription not available"):
        transcriber.transcribe("/tmp/audio.ogg")


def test_load_raises_import_error_when_whisper_not_installed():
    import transcriber
    transcriber._model = None

    with patch.dict(sys.modules, {"whisper": None}):
        with pytest.raises(ImportError):
            transcriber.load()


def test_load_is_idempotent():
    import transcriber
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        transcriber.load()
        transcriber.load()  # second call should be a no-op

    mock_whisper.load_model.assert_called_once_with("base")  # only called once
