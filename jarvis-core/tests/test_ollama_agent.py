import json
import pytest
from unittest.mock import MagicMock, patch
from guardrails import Guardrails
from ollama_agent import OllamaAgent, EscalateToCloud, _anthropic_to_ollama_tools


# ── helper: build a fake Ollama HTTP response ─────────────────────────────────

def _stop_response(text: str) -> MagicMock:
    """Simulate Ollama returning a final text answer."""
    msg = MagicMock()
    msg.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": text, "tool_calls": None},
                     "finish_reason": "stop"}]
    }
    return msg


def _tool_response(tool_name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    """Simulate Ollama calling a tool."""
    msg = MagicMock()
    msg.json.return_value = {
        "choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}}]
        }, "finish_reason": "tool_calls"}]
    }
    return msg


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    cfg = config.load()
    guardrails = Guardrails(cfg)
    return OllamaAgent(config=cfg, guardrails=guardrails)


# ── tool schema translation ───────────────────────────────────────────────────

def test_anthropic_to_ollama_tools_converts_schema():
    anthropic_tools = [{
        "name": "shell_run",
        "description": "Run a shell command",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
    }]
    result = _anthropic_to_ollama_tools(anthropic_tools)
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "shell_run"
    assert result[0]["function"]["parameters"]["properties"]["command"]["type"] == "string"


# ── happy path: Ollama returns final answer immediately ───────────────────────

def test_run_returns_response_when_ollama_answers(agent):
    with patch("httpx.Client.post", return_value=_stop_response("Your Downloads are empty.")):
        result = agent.run("list my Downloads", cwd=None)
    assert result["speak"] == "Your Downloads are empty."
    assert result["display"] == "Your Downloads are empty."
    assert result.get("error") is None


# ── tool use: Ollama calls shell_run then answers ─────────────────────────────

def test_run_executes_tool_and_returns_response(agent):
    responses = [
        _tool_response("shell_run", {"command": "ls ~/Downloads"}, "call_1"),
        _stop_response("You have 3 files in Downloads."),
    ]
    with patch("httpx.Client.post", side_effect=responses):
        with patch.object(agent._shell, "run", return_value={"exit_code": 0, "stdout": "a.txt\nb.txt\nc.txt", "stderr": ""}):
            result = agent.run("list my Downloads", cwd=None)
    assert result["speak"] == "You have 3 files in Downloads."


# ── ollama_only mode: no escalation raised, just max-iter response ────────────

def test_run_in_ollama_only_mode_does_not_raise_escalate(agent):
    agent._config["ollama"]["routing_mode"] = "ollama_only"
    responses = [
        _tool_response("escalate_to_claude", {"reason": "requires web search"}, "call_1"),
        _stop_response("I cannot search the web in offline mode."),
    ]
    with patch("httpx.Client.post", side_effect=responses):
        result = agent.run("what is the latest React version", cwd=None)
    assert result["speak"]  # returned something instead of raising


# ── connection error: Ollama not running ──────────────────────────────────────

def test_run_raises_escalate_when_ollama_unreachable(agent):
    import httpx
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("connection refused")):
        with pytest.raises(EscalateToCloud) as exc_info:
            agent.run("open Safari", cwd=None)
    assert "unavailable" in exc_info.value.reason.lower() or "connect" in exc_info.value.reason.lower()


# ── timeout: Ollama too slow (model loading) ─────────────────────────────────

def test_run_raises_escalate_when_ollama_times_out(agent):
    import httpx
    with patch("httpx.Client.post", side_effect=httpx.ReadTimeout("timed out")):
        with pytest.raises(EscalateToCloud) as exc_info:
            agent.run("run tests", cwd=None)
    assert "timeout" in exc_info.value.reason.lower()


# ── malformed JSON in tool arguments ─────────────────────────────────────────

def test_run_recovers_from_malformed_tool_arguments(agent):
    """If Ollama emits invalid JSON in tool arguments, loop sends error result back and continues."""
    bad_tc_response = MagicMock()
    bad_tc_response.json.return_value = {
        "choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_bad", "type": "function",
                            "function": {"name": "shell_run", "arguments": "not-valid-json{"}}]
        }, "finish_reason": "tool_calls"}]
    }
    responses = [bad_tc_response, _stop_response("Sorry, I had a formatting error. Here is the result.")]
    with patch("httpx.Client.post", side_effect=responses):
        result = agent.run("list files", cwd=None)
    # Should not crash — returns a response from the second iteration
    assert result["speak"]


# ── HTTP error: Ollama returns 503 (model not loaded) ─────────────────────────

def test_run_raises_escalate_when_ollama_returns_http_error(agent):
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable", request=MagicMock(), response=mock_resp
    )
    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(EscalateToCloud) as exc_info:
            agent.run("run my tests", cwd=None)
    assert "http error" in exc_info.value.reason.lower()


# ── guardrails: approval required ─────────────────────────────────────────────

def test_run_returns_approval_required_when_guardrails_block(agent):
    agent._config["guardrails"]["delete_files"] = "require_approval"
    responses = [
        _tool_response("shell_run", {"command": "rm -rf /tmp/foo"}, "call_1"),
    ]
    # Simulate guardrails blocking the shell_run tool
    with patch("httpx.Client.post", side_effect=responses):
        with patch.object(agent._guardrails, "classify") as mock_classify:
            from guardrails import Decision
            mock_classify.return_value = Decision.REQUIRE_APPROVAL
            result = agent.run("delete /tmp/foo", cwd=None)
    assert result["speak"] is None
    assert result["display"] is None
    assert "approval_required" in result
    assert result["approval_required"]["tool_use_id"] == "call_1"
    assert "steps" in result   # consistent response shape with Agent.run()


# ── delegate_to_claude_code is not offered to Ollama ─────────────────────────

def test_ollama_tools_exclude_delegate_to_claude_code():
    from ollama_agent import _OLLAMA_TOOLS
    names = [t["function"]["name"] for t in _OLLAMA_TOOLS]
    assert "delegate_to_claude_code" not in names
    assert "escalate_to_claude" not in names   # removed: pre-flight classifier handles routing
    assert "delegate_to_local" not in names    # Claude-only tool: Ollama is the delegate, not the delegator


def test_run_injects_memory_context_into_messages(agent):
    """If memory_context is provided, it appears in the system message."""
    captured_messages = []

    def fake_post(url, **kwargs):
        captured_messages.extend(kwargs.get("json", {}).get("messages", []))
        return _stop_response("Done.")

    with patch("httpx.Client.post", side_effect=fake_post):
        agent.run("run tests", cwd="/my/project", memory_context="Test: npm test")

    system_msg = next((m for m in captured_messages if m["role"] == "system"), None)
    assert system_msg is not None
    assert "npm test" in system_msg["content"]
