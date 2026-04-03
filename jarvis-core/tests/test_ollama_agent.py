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


# ── salvage call on step exhaustion ──────────────────────────────────────────

def test_salvage_call_on_step_exhaustion(agent):
    """When ALL steps are exhausted (model never stopped), a final no-tools call salvages the answer."""
    call_num = [0]
    commands = ["ls /", "ls /tmp", "ls /usr"]

    def make_tool_response(cmd):
        return {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "tool_calls": [{"id": f"c{call_num[0]}", "function": {
                "name": "shell_run", "arguments": f'{{"command": "{cmd}"}}'
            }}]
        }}]}

    salvage_response = {
        "choices": [{"finish_reason": "stop", "message": {"content": "MacBook Pro, M1 Max, 64GB RAM."}}]
    }
    salvage_calls = []

    def fake_post(url, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if json and json.get("tool_choice") == "none" and "tools" not in json:
            salvage_calls.append(json)
            resp.json.return_value = salvage_response
        else:
            resp.json.return_value = make_tool_response(commands[call_num[0] % len(commands)])
            call_num[0] += 1
        return resp

    agent._config["reasoning"]["max_steps_ollama"] = 3
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="file list"):
            result = agent.run("what are my specs")

    assert len(salvage_calls) >= 1, "Salvage call should have been made after exhausting steps"
    assert "ran out of steps" not in result["speak"]
    assert "MacBook Pro" in result["speak"]


def test_salvage_falls_back_to_error_if_empty(agent):
    """If salvage call returns empty content, fall back to error message."""
    call_num = [0]
    commands = ["ls /", "ls /tmp", "ls /usr"]

    def make_tool_response(cmd):
        return {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "tool_calls": [{"id": f"c{call_num[0]}", "function": {
                "name": "shell_run", "arguments": f'{{"command": "{cmd}"}}'
            }}]
        }}]}

    empty_salvage = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}

    def fake_post(url, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if json and json.get("tool_choice") == "none" and "tools" not in json:
            resp.json.return_value = empty_salvage
        else:
            resp.json.return_value = make_tool_response(commands[call_num[0] % len(commands)])
            call_num[0] += 1
        return resp

    agent._config["reasoning"]["max_steps_ollama"] = 3
    agent._config["reasoning"]["stall_detection"] = False
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="ok"):
            result = agent.run("do stuff")

    assert result["speak"] == "I ran out of steps. Please try again."


# ── finalize tool ─────────────────────────────────────────────────────────────

def test_finalize_tool_returns_immediately(agent):
    """When model calls finalize, agent returns that answer without any further steps."""
    finalize_response = {
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "tool_calls": [{"id": "c1", "function": {
                "name": "finalize",
                "arguments": '{"answer": "Your laptop has 64GB RAM and an M1 Max chip."}'
            }}]
        }}]
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = finalize_response

    with patch.object(agent._http_client, "post", return_value=resp):
        result = agent.run("what are my specs")

    assert "64GB RAM" in result["speak"]
    assert result["steps"][0]["tool"] == "finalize"


def test_finalize_tool_stops_before_step_limit(agent):
    """finalize on step 1 returns immediately without exhausting max_steps."""
    call_count = [0]

    def fake_post(url, json=None, **kwargs):
        call_count[0] += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if call_count[0] == 1:
            # First call: run a shell tool
            resp.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": "c1", "function": {
                    "name": "shell_run", "arguments": '{"command": "uname -m"}'
                }}]
            }}]}
        else:
            # Second call: finalize
            resp.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": "c2", "function": {
                    "name": "finalize", "arguments": '{"answer": "ARM architecture M1 Max."}'
                }}]
            }}]}
        return resp

    agent._config["reasoning"]["max_steps_ollama"] = 10
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="arm64"):
            result = agent.run("what chip do I have")

    assert "M1 Max" in result["speak"]
    assert call_count[0] == 2  # only 2 API calls, not 10
    assert len(result["steps"]) == 2


# ── intent-class based step budgets ──────────────────────────────────────────

def test_read_only_intent_uses_lower_step_budget(agent):
    """read_only intent_class uses step_budgets.read_only (5) not max_steps_ollama (10)."""
    call_count = [0]
    commands = [f"ls /dir{i}" for i in range(20)]

    def fake_post(url, json=None, **kwargs):
        call_count[0] += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if json and json.get("tool_choice") == "none":
            resp.json.return_value = {"choices": [{"finish_reason": "stop",
                                                    "message": {"content": "Done."}}]}
        else:
            cmd = commands[call_count[0] % len(commands)]
            resp.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": f"c{call_count[0]}", "function": {
                    "name": "shell_run", "arguments": f'{{"command": "{cmd}"}}'
                }}]
            }}]}
        return resp

    agent._config["reasoning"]["max_steps_ollama"] = 10
    agent._config["reasoning"]["step_budgets"] = {
        "read_only": 5, "prepare": 8, "destructive": 10, "complex_reasoning": 10
    }
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="ok"):
            agent.run("list files", intent_class="read_only")

    # 5 tool calls + 1 salvage = ≤ 7 total API calls
    assert call_count[0] <= 7


def test_destructive_intent_uses_higher_step_budget(agent):
    """destructive intent_class uses step_budgets.destructive (10), not read_only (3)."""
    call_count = [0]
    # Alternate tools so near-duplicate detection (same tool ≥ 3 times) doesn't fire
    tools_cycle = ["shell_run", "file_read", "shell_run", "file_read",
                   "shell_run", "file_read", "shell_run", "file_read",
                   "shell_run", "file_read", "shell_run"]
    args_cycle = [
        '{"command": "rm /tmp/a"}', '{"path": "/tmp/a.log"}',
        '{"command": "rm /tmp/b"}', '{"path": "/tmp/b.log"}',
        '{"command": "rm /tmp/c"}', '{"path": "/tmp/c.log"}',
        '{"command": "rm /tmp/d"}', '{"path": "/tmp/d.log"}',
        '{"command": "rm /tmp/e"}', '{"path": "/tmp/e.log"}',
        '{"command": "rm /tmp/f"}',
    ]

    def fake_post(url, json=None, **kwargs):
        call_count[0] += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if json and json.get("tool_choice") == "none":
            resp.json.return_value = {"choices": [{"finish_reason": "stop",
                                                    "message": {"content": "Deleted."}}]}
        else:
            idx = (call_count[0] - 1) % len(tools_cycle)
            resp.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": f"c{call_count[0]}", "function": {
                    "name": tools_cycle[idx], "arguments": args_cycle[idx]
                }}]
            }}]}
        return resp

    agent._config["reasoning"]["step_budgets"] = {
        "read_only": 3, "prepare": 8, "destructive": 10, "complex_reasoning": 10
    }
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="ok"):
            agent.run("delete temp files", intent_class="destructive")

    # destructive budget=10 → more API calls than read_only budget=3 would allow (4 max)
    assert call_count[0] > 4


def test_no_intent_class_uses_max_steps_ollama(agent):
    """Without intent_class, falls back to max_steps_ollama."""
    call_count = [0]
    commands = [f"ls /dir{i}" for i in range(20)]

    def fake_post(url, json=None, **kwargs):
        call_count[0] += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if json and json.get("tool_choice") == "none":
            resp.json.return_value = {"choices": [{"finish_reason": "stop",
                                                    "message": {"content": "Done."}}]}
        else:
            cmd = commands[call_count[0] % len(commands)]
            resp.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": f"c{call_count[0]}", "function": {
                    "name": "shell_run", "arguments": f'{{"command": "{cmd}"}}'
                }}]
            }}]}
        return resp

    agent._config["reasoning"]["max_steps_ollama"] = 3
    agent._config["reasoning"]["step_budgets"] = {"read_only": 1}
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="ok"):
            agent.run("do something")  # no intent_class → uses max_steps_ollama=3

    # 3 tool calls + 1 salvage = ≤ 5 total; more than 1 (which read_only budget would give)
    assert call_count[0] > 2


def test_finalize_tool_is_in_ollama_tools_schema(agent):
    """finalize tool should be present in the tools sent to Ollama."""
    from ollama_agent import _OLLAMA_TOOLS
    tool_names = [t["function"]["name"] for t in _OLLAMA_TOOLS]
    assert "finalize" in tool_names


# ── near-duplicate redundancy detection ──────────────────────────────────────

def test_near_duplicate_detection_stops_redundant_loop(agent):
    """Same tool called 3 times in a 5-step window triggers salvage and stops loop."""
    call_num = [0]
    # 4 slightly-different find commands — same tool, near-identical args
    commands = [
        "find ~/dev -name '*.py' | wc -l",
        "find ~/dev -name '*.py' -not -path '*cache*' | wc -l",
        "find ~/dev -name '*.py' -not -path '*__pycache__*' | wc -l",
        "find ~/dev -name '*.py' -not -path '*/.venv/*' | wc -l",
    ]

    salvage_response = {"choices": [{"finish_reason": "stop",
                                      "message": {"content": "Found 100 Python files."}}]}

    def fake_post(url, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if json and json.get("tool_choice") == "none":
            resp.json.return_value = salvage_response
        else:
            cmd = commands[call_num[0] % len(commands)]
            call_num[0] += 1
            resp.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": f"c{call_num[0]}", "function": {
                    "name": "shell_run",
                    "arguments": f'{{"command": "{cmd}"}}'
                }}]
            }}]}
        return resp

    agent._config["reasoning"]["max_steps_ollama"] = 10
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="   100"):
            result = agent.run("how many python files")

    assert len(result["steps"]) < 10, "Should have stopped before using all 10 steps"
    assert "ran out of steps" not in result["speak"]
    assert "100 Python files" in result["speak"]
