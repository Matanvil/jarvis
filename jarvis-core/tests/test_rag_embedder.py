import pytest
from unittest.mock import MagicMock, patch
from rag.embedder import OllamaEmbedder, EmbedderError


def test_embed_returns_float_list():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.Client.post", return_value=mock_resp):
        embedder = OllamaEmbedder()
        result = embedder.embed("hello world")
    assert result == [0.1, 0.2, 0.3]


def test_embed_truncates_long_text():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.5]}
    mock_resp.raise_for_status = MagicMock()
    captured = {}

    def fake_post(url, json, **kwargs):
        captured["prompt"] = json["prompt"]
        return mock_resp

    with patch("httpx.Client.post", side_effect=fake_post):
        embedder = OllamaEmbedder()
        embedder.embed("x" * 10000)
    assert len(captured["prompt"]) == 6000


def test_embed_raises_on_missing_embedding_key():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"error": "model not found"}
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.Client.post", return_value=mock_resp):
        embedder = OllamaEmbedder()
        with pytest.raises(EmbedderError, match="Unexpected response"):
            embedder.embed("test")


def test_embed_raises_on_connect_error():
    import httpx
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("refused")):
        embedder = OllamaEmbedder()
        with pytest.raises(EmbedderError, match="not running"):
            embedder.embed("test")


def test_embed_raises_on_timeout():
    import httpx
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("timeout")):
        embedder = OllamaEmbedder()
        with pytest.raises(EmbedderError, match="timed out"):
            embedder.embed("test")
