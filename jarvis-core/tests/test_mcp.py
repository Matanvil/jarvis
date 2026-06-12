"""Tests for MCPManager — tools/mcp.py"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mcp_tool(name: str, description: str = "", props: dict | None = None, required: list | None = None):
    """Return a mock MCP Tool object."""
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = {
        "type": "object",
        "properties": props or {"input": {"type": "string"}},
        "required": required or [],
    }
    return t


def _make_text_content(text: str):
    c = MagicMock()
    c.type = "text"
    c.text = text
    return c


def _server_cfg(name="fs", command="npx", args=None, env=None, transport="stdio"):
    return {"name": name, "command": command, "args": args or [], "env": env, "transport": transport}


# ---------------------------------------------------------------------------
# MCPManager construction
# ---------------------------------------------------------------------------

def test_mcp_manager_initialises_empty():
    from tools.mcp import MCPManager
    mgr = MCPManager([])
    assert mgr.tool_schemas() == []
    assert mgr.server_names() == []


def test_mcp_manager_stores_server_configs():
    from tools.mcp import MCPManager
    servers = [_server_cfg("fs"), _server_cfg("gh")]
    mgr = MCPManager(servers)
    assert mgr.server_names() == ["fs", "gh"]


# ---------------------------------------------------------------------------
# tool_schemas — prefix + OpenAI shape
# ---------------------------------------------------------------------------

def test_tool_schemas_empty_before_connect():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])
    assert mgr.tool_schemas() == []


def test_tool_schemas_after_inject_returns_prefixed_tools():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])
    tools = [_make_mcp_tool("read_file", "Read a file", {"path": {"type": "string"}}, ["path"])]
    mgr._inject_tools("fs", tools)
    schemas = mgr.tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "mcp__fs__read_file"
    assert schemas[0]["description"] == "Read a file"
    assert "path" in schemas[0]["input_schema"]["properties"]


def test_tool_schemas_openai_shape():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("gh")])
    tools = [_make_mcp_tool("list_repos", "List GitHub repos")]
    mgr._inject_tools("gh", tools)
    schema = mgr.tool_schemas()[0]
    assert "name" in schema
    assert "description" in schema
    assert "input_schema" in schema
    assert schema["input_schema"]["type"] == "object"


def test_tool_schemas_multiple_servers_all_returned():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs"), _server_cfg("gh")])
    mgr._inject_tools("fs", [_make_mcp_tool("read_file")])
    mgr._inject_tools("gh", [_make_mcp_tool("list_repos"), _make_mcp_tool("create_issue")])
    schemas = mgr.tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"mcp__fs__read_file", "mcp__gh__list_repos", "mcp__gh__create_issue"}


# ---------------------------------------------------------------------------
# is_mcp_tool / parse_mcp_tool_name
# ---------------------------------------------------------------------------

def test_is_mcp_tool_true_for_prefixed():
    from tools.mcp import MCPManager
    mgr = MCPManager([])
    assert mgr.is_mcp_tool("mcp__fs__read_file") is True


def test_is_mcp_tool_false_for_builtin():
    from tools.mcp import MCPManager
    mgr = MCPManager([])
    assert mgr.is_mcp_tool("shell_run") is False
    assert mgr.is_mcp_tool("file_read") is False


def test_parse_mcp_tool_name():
    from tools.mcp import MCPManager
    mgr = MCPManager([])
    server, tool = mgr.parse_mcp_tool_name("mcp__gh__create_issue")
    assert server == "gh"
    assert tool == "create_issue"


def test_parse_mcp_tool_name_with_double_underscore_in_tool():
    from tools.mcp import MCPManager
    mgr = MCPManager([])
    server, tool = mgr.parse_mcp_tool_name("mcp__fs__read_file_contents")
    assert server == "fs"
    assert tool == "read_file_contents"


# ---------------------------------------------------------------------------
# call_tool — sync wrapper around async session
# ---------------------------------------------------------------------------

def test_call_tool_returns_text_content():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])

    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = [_make_text_content("file contents here")]

    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    mgr._sessions["fs"] = mock_session

    result = mgr.call_tool("fs", "read_file", {"path": "/tmp/test.txt"})
    assert result == "file contents here"


def test_call_tool_joins_multiple_content_blocks():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])

    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = [_make_text_content("part1"), _make_text_content("part2")]

    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    mgr._sessions["fs"] = mock_session

    result = mgr.call_tool("fs", "read_file", {"path": "/tmp/test.txt"})
    assert "part1" in result
    assert "part2" in result


def test_call_tool_returns_error_string_on_is_error():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])

    mock_result = MagicMock()
    mock_result.isError = True
    mock_result.content = [_make_text_content("permission denied")]

    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    mgr._sessions["fs"] = mock_session

    result = mgr.call_tool("fs", "read_file", {"path": "/etc/shadow"})
    assert "error" in result.lower() or "permission denied" in result


def test_call_tool_raises_when_server_not_connected():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])
    # no session injected
    with pytest.raises(RuntimeError, match="not connected"):
        mgr.call_tool("fs", "read_file", {"path": "/tmp/test.txt"})


def test_call_tool_unknown_server_raises():
    from tools.mcp import MCPManager
    mgr = MCPManager([])
    with pytest.raises(RuntimeError):
        mgr.call_tool("nonexistent", "some_tool", {})


# ---------------------------------------------------------------------------
# start_all / stop_all
# ---------------------------------------------------------------------------

def test_start_all_skips_empty_server_list():
    from tools.mcp import MCPManager
    mgr = MCPManager([])
    mgr.start_all()  # should not raise
    assert mgr.tool_schemas() == []


def test_start_all_logs_warning_on_server_failure(caplog):
    from tools.mcp import MCPManager
    import logging
    servers = [_server_cfg("bad", command="nonexistent-binary-xyz")]
    mgr = MCPManager(servers)
    with caplog.at_level(logging.WARNING):
        mgr.start_all()  # bad binary — should warn, not raise
    assert mgr.tool_schemas() == []


def test_stop_all_clears_sessions():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])
    mock_session = AsyncMock()
    mgr._sessions["fs"] = mock_session
    mgr._inject_tools("fs", [_make_mcp_tool("read_file")])

    mgr.stop_all()
    assert mgr._sessions == {}
    assert mgr.tool_schemas() == []


# ---------------------------------------------------------------------------
# Guardrail category
# ---------------------------------------------------------------------------

def test_guardrail_category_defaults_to_mcp_tool():
    from tools.mcp import MCPManager
    mgr = MCPManager([_server_cfg("fs")])
    assert mgr.guardrail_category("fs") == "mcp_tool"


def test_guardrail_category_respects_config_override():
    from tools.mcp import MCPManager
    servers = [_server_cfg("fs") | {"guardrail": "auto_allow"}]
    mgr = MCPManager(servers)
    assert mgr.guardrail_category("fs") == "auto_allow"
