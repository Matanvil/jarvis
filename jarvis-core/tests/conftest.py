import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import httpx


@pytest.fixture(autouse=True)
def no_dpo_writes(tmp_path):
    """Redirect DPO captures to a temp dir so tests never write to ~/.jarvis/logs/."""
    with patch("local_agent.DPO_LOG_PATH", str(tmp_path / "dpo_data.jsonl")):
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
