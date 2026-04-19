import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import httpx


@pytest.fixture(autouse=True)
def no_dev_config():
    """Prevent config.dev.json on disk from affecting any test."""
    with patch("config.DEV_CONFIG_PATH", Path("/nonexistent/config.dev.json")):
        yield


@pytest.fixture(autouse=True)
def no_real_http_stream():
    """Prevent httpx.Client.stream from making real network calls in tests.

    Tests that want to exercise streaming must use patch.object(agent._http_client, "stream", ...)
    which takes precedence over this class-level patch.  All other tests hit the _stream_call
    fallback path which calls .post() — that is already mocked in those tests.
    """
    with patch.object(httpx.Client, "stream", side_effect=Exception("stream not mocked in this test")):
        yield
