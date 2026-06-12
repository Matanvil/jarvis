import pytest
import httpx
from unittest.mock import patch, MagicMock
from tools.web import WebTool


def test_search_returns_results_with_content():
    tool = WebTool(brave_api_key=None)
    mock_response = MagicMock()
    mock_response.text = """
    <html><body>
    <div class="result__title"><a href="https://example.com">Example Title</a></div>
    <div class="result__snippet">Example snippet text</div>
    </body></html>
    """
    mock_response.raise_for_status = MagicMock()
    with patch("tools.web.httpx.get", return_value=mock_response):
        results = tool.search("python fastapi tutorial")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0]["title"] == "Example Title"
    assert results[0]["url"] == "https://example.com"
    assert results[0]["snippet"] == "Example snippet text"


def test_search_handles_network_error():
    tool = WebTool(brave_api_key=None)
    with patch("tools.web.httpx.get", side_effect=Exception("network error")):
        results = tool.search("anything")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["error"] is not None


def test_fetch_page_extracts_text():
    tool = WebTool(brave_api_key=None)
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello world</p><p>More content here</p></body></html>"
    mock_response.raise_for_status = MagicMock()
    with patch("tools.web.httpx.get", return_value=mock_response):
        result = tool.fetch_page("https://example.com")
    assert "Hello world" in result["text"]
    assert result["error"] is None


def test_fetch_page_handles_error():
    tool = WebTool(brave_api_key=None)
    with patch("tools.web.httpx.get", side_effect=Exception("connection refused")):
        result = tool.fetch_page("https://notreal.invalid")
    assert result["error"] is not None
    assert result["text"] is None


def test_fetch_page_default_max_chars_is_20000():
    """fetch_page default limit should be 20 000 chars, not 4 000."""
    tool = WebTool(brave_api_key=None)
    import inspect
    sig = inspect.signature(tool.fetch_page)
    assert sig.parameters["max_chars"].default == 20000


def test_web_tool_uses_httpx_not_requests():
    """WebTool should use httpx.get, not requests.get, for HTTP calls."""
    tool = WebTool(brave_api_key=None)
    mock_response = MagicMock()
    mock_response.text = """
    <html><body>
    <div class="result__title"><a href="https://example.com">Title</a></div>
    <div class="result__snippet">Snippet text</div>
    </body></html>
    """
    mock_response.raise_for_status = MagicMock()
    with patch("tools.web.httpx.get", return_value=mock_response) as mock_get:
        results = tool.search("test query")
    mock_get.assert_called_once()


def test_fetch_page_uses_httpx():
    """fetch_page should use httpx.get."""
    tool = WebTool(brave_api_key=None)
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello</p></body></html>"
    mock_response.raise_for_status = MagicMock()
    with patch("tools.web.httpx.get", return_value=mock_response) as mock_get:
        result = tool.fetch_page("https://example.com")
    mock_get.assert_called_once()
    assert result["error"] is None
