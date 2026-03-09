import pytest
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
    with patch("tools.web.requests.get", return_value=mock_response):
        results = tool.search("python fastapi tutorial")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0]["title"] == "Example Title"
    assert results[0]["url"] == "https://example.com"


def test_search_handles_network_error():
    tool = WebTool(brave_api_key=None)
    with patch("tools.web.requests.get", side_effect=Exception("network error")):
        results = tool.search("anything")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["error"] is not None


def test_fetch_page_extracts_text():
    tool = WebTool(brave_api_key=None)
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello world</p><p>More content here</p></body></html>"
    mock_response.raise_for_status = MagicMock()
    with patch("tools.web.requests.get", return_value=mock_response):
        result = tool.fetch_page("https://example.com")
    assert "Hello world" in result["text"]
    assert result["error"] is None


def test_fetch_page_handles_error():
    tool = WebTool(brave_api_key=None)
    with patch("tools.web.requests.get", side_effect=Exception("connection refused")):
        result = tool.fetch_page("https://notreal.invalid")
    assert result["error"] is not None
    assert result["text"] is None
