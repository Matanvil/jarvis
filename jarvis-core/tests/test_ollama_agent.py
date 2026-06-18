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


# ── max_steps_ollama controls step limit ──────────────────────────────────────

def test_max_steps_ollama_controls_step_limit(agent):
    """max_steps_ollama limits total tool calls regardless of intent_class."""
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
    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", return_value="ok"):
            agent.run("do something")

    # 3 tool calls + salvage = ≤ 5 total API calls
    assert call_count[0] <= 5


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


# ── coding agent wiring ───────────────────────────────────────────────────────

def test_ollama_agent_has_coding_agent(agent):
    """OllamaAgent should have a _coding attribute (CodingAgentTool instance)."""
    assert hasattr(agent, "_coding")
    assert agent._coding is not None


def test_coding_ask_tool_routes_to_coding_agent(agent):
    """coding_ask tool call should invoke the coding agent, not return 'not available'."""
    coding_response = _tool_response("coding_ask", {"question": "How does routing work?", "cwd": "/some/project"})
    stop = _stop_response("Routing uses a pre-flight classifier.")

    responses = [coding_response, stop]
    call_idx = [0]

    def fake_post(url, json=None, **kwargs):
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    mock_coding = MagicMock()
    mock_coding.ask.return_value = {"answer": "Routing uses a pre-flight classifier.", "error": None}
    agent._coding = mock_coding

    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", wraps=lambda name, args, *a, **kw: (
            mock_coding.ask(args["question"], args["cwd"])["answer"]
            if name == "coding_ask" else "ok"
        )):
            result = agent.run("How does routing work?", cwd="/some/project")

    mock_coding.ask.assert_called_once()


def test_coding_tools_in_ollama_schema():
    """coding_ask, coding_plan, coding_review should be in OllamaAgent's tool schema."""
    from ollama_agent import _OLLAMA_TOOLS
    tool_names = [t["function"]["name"] for t in _OLLAMA_TOOLS]
    assert "coding_ask" in tool_names
    assert "coding_plan" in tool_names
    assert "coding_review" in tool_names


def test_http_client_uses_split_timeout(agent):
    """OllamaAgent._http_client must use httpx.Timeout with short connect and long read."""
    import httpx
    t = agent._http_client.timeout
    assert isinstance(t, httpx.Timeout), "Expected httpx.Timeout, not a flat number"
    assert t.connect <= 10.0, f"Connect timeout should be ≤10s, got {t.connect}"
    assert t.read >= 60.0, f"Read timeout should be ≥60s, got {t.read}"


def test_timeout_seconds_config_sets_read_timeout(tmp_path, monkeypatch):
    """timeout_seconds in config should set the read timeout, not the connect timeout."""
    import httpx
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    cfg = config.load()
    cfg["ollama"]["timeout_seconds"] = 120
    from guardrails import Guardrails
    a = OllamaAgent(config=cfg, guardrails=Guardrails(cfg))
    assert a._http_client.timeout.read == 120.0
    assert a._http_client.timeout.connect <= 10.0


def test_step_callback_fired_for_all_steps_not_just_first(agent):
    """step_callback must be called for every tool call, not just the first (milestone)."""
    responses = [
        _tool_response("shell_run", {"command": "ls /"}, "call_1"),
        _tool_response("shell_run", {"command": "ls /tmp"}, "call_2"),
        _stop_response("Done."),
    ]
    step_events = []
    with patch("httpx.Client.post", side_effect=responses):
        with patch("ollama_agent.execute_tool", return_value="ok"):
            agent.run("list dirs", step_callback=lambda e: step_events.append(e))

    step_type_events = [e for e in step_events if e["type"] == "step"]
    assert len(step_type_events) == 2, f"Expected 2 step events, got {len(step_type_events)}"
    assert step_type_events[0]["milestone"] is True   # first step is milestone
    assert step_type_events[1]["milestone"] is False  # second step is not


def test_non_milestone_step_event_has_correct_fields(agent):
    """Non-milestone step events must include type, label, tool, milestone=False."""
    responses = [
        _tool_response("shell_run", {"command": "ls /"}, "call_1"),
        _tool_response("file_read", {"path": "/tmp/foo"}, "call_2"),
        _stop_response("Done."),
    ]
    step_events = []
    with patch("httpx.Client.post", side_effect=responses):
        with patch("ollama_agent.execute_tool", return_value="ok"):
            agent.run("do stuff", step_callback=lambda e: step_events.append(e))

    non_milestones = [e for e in step_events if e.get("type") == "step" and not e["milestone"]]
    assert len(non_milestones) == 1
    e = non_milestones[0]
    assert e["type"] == "step"
    assert "label" in e
    assert e["tool"] == "file_read"
    assert e["milestone"] is False


from ollama_agent import _stream_call
import json as _json_mod


def _make_streaming_tool_response(tool_name: str, args: dict, call_id: str = "call_1"):
    """Build list of SSE line strings simulating a streaming tool call response."""
    args_str = _json_mod.dumps(args)
    mid = len(args_str) // 2
    lines = [
        f'data: {_json_mod.dumps({"choices": [{"delta": {"role": "assistant", "content": "", "tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": tool_name, "arguments": ""}}]}, "finish_reason": None}]})}',
        f'data: {_json_mod.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_str[:mid]}}]}, "finish_reason": None}]})}',
        f'data: {_json_mod.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_str[mid:]}}]}, "finish_reason": None}]})}',
        f'data: {_json_mod.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})}',
        "data: [DONE]",
    ]
    return lines


def _make_streaming_text_response(text: str):
    """Build list of SSE line strings simulating a streaming text response."""
    lines = []
    for char in text:
        lines.append(f'data: {_json_mod.dumps({"choices": [{"delta": {"content": char}, "finish_reason": None}]})}')
    lines.append(f'data: {_json_mod.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}')
    lines.append("data: [DONE]")
    return lines


def _mock_stream(lines):
    """Return a mock context manager that yields the given SSE lines."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(lines))
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


def test_stream_call_accumulates_tool_call_fragments(agent):
    """_stream_call must reassemble fragmented tool call arguments into a valid message."""
    lines = _make_streaming_tool_response("shell_run", {"command": "ls -la"}, "call_abc")
    with patch.object(agent._http_client, "stream", return_value=_mock_stream(lines)):
        msg, finish = _stream_call(agent._http_client, "http://localhost/v1/chat/completions",
                                   {"model": "test", "messages": [], "tools": []}, step_callback=None)

    assert finish == "tool_calls"
    assert msg["tool_calls"] is not None
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "shell_run"
    parsed = _json_mod.loads(tc["function"]["arguments"])
    assert parsed["command"] == "ls -la"


def test_stream_call_skips_role_only_chunks(agent):
    """_stream_call must ignore role-only opener chunks when detecting branch."""
    lines = [
        'data: {"choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]}',
        *_make_streaming_tool_response("shell_run", {"command": "pwd"}, "call_x"),
    ]
    with patch.object(agent._http_client, "stream", return_value=_mock_stream(lines)):
        msg, finish = _stream_call(agent._http_client, "http://localhost/v1/chat/completions",
                                   {"model": "test", "messages": [], "tools": []}, step_callback=None)
    assert msg["tool_calls"] is not None
    assert msg["tool_calls"][0]["function"]["name"] == "shell_run"


def test_stream_call_falls_back_to_non_streaming_on_parse_error(agent):
    """If streaming raises a non-httpx error (e.g. ValueError), _stream_call retries with stream=False."""
    non_streaming_resp = MagicMock()
    non_streaming_resp.raise_for_status = MagicMock()
    non_streaming_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "fallback answer", "tool_calls": None},
                     "finish_reason": "stop"}]
    }
    with patch.object(agent._http_client, "stream", side_effect=ValueError("unexpected stream format")):
        with patch.object(agent._http_client, "post", return_value=non_streaming_resp):
            msg, finish = _stream_call(agent._http_client, "http://localhost/v1/chat/completions",
                                       {"model": "test", "messages": [], "tools": []}, step_callback=None)
    assert msg["content"] == "fallback answer"
    assert finish == "stop"


def test_stream_call_reraises_httpx_timeout(agent):
    """httpx.TimeoutException from streaming must propagate up, not fall back to non-streaming."""
    import httpx
    with patch.object(agent._http_client, "stream", side_effect=httpx.ReadTimeout("timed out")):
        with pytest.raises(httpx.TimeoutException):
            _stream_call(agent._http_client, "http://localhost/v1/chat/completions",
                         {"model": "test", "messages": [], "tools": []}, step_callback=None)


def test_run_fires_token_events_for_text_response(agent):
    """When Ollama streams a text answer, run() must forward token events via step_callback."""
    text = "Your system has 64GB RAM."
    lines = _make_streaming_text_response(text)

    events = []
    with patch.object(agent._http_client, "stream", return_value=_mock_stream(lines)):
        agent.run("what are my specs", step_callback=lambda e: events.append(e))

    token_events = [e for e in events if e.get("type") == "token"]
    assert len(token_events) > 0, "Expected token events, got none"
    assembled = "".join(e["text"] for e in token_events)
    assert assembled == text


def test_run_no_spurious_token_events_during_tool_call_round(agent):
    """Tool call rounds must not emit token events — only step events for that round."""
    tool_lines = _make_streaming_tool_response("shell_run", {"command": "ls"}, "call_1")
    stop_lines = _make_streaming_text_response("Done.")

    events = []
    call_count = [0]

    def fake_stream(method, url, json=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_stream(tool_lines)
        return _mock_stream(stop_lines)

    with patch.object(agent._http_client, "stream", side_effect=fake_stream):
        with patch("ollama_agent.execute_tool", return_value="file list"):
            agent.run("list files", step_callback=lambda e: events.append(e))

    # Verify zero tokens emitted during the tool call round (before first step event)
    first_step_idx = next((i for i, e in enumerate(events) if e.get("type") == "step"), len(events))
    pre_step_tokens = [e for e in events[:first_step_idx] if e.get("type") == "token"]
    assert pre_step_tokens == [], f"Tool call round emitted spurious tokens: {pre_step_tokens}"

    # All token events should come from the final text round only (not the tool call round)
    token_events = [e for e in events if e.get("type") == "token"]
    assert len(token_events) > 0, "Expected token events from final text response"
    assembled = "".join(e["text"] for e in token_events)
    assert assembled == "Done."
    # Also verify there was at least one step event (from the tool call)
    step_events = [e for e in events if e.get("type") == "step"]
    assert len(step_events) >= 1


def test_coding_agent_passed_to_execute_tool(agent):
    """execute_tool should be called with the coding agent instance."""
    coding_response = _tool_response("coding_ask", {"question": "What is agent.py?", "cwd": "/p"})
    stop = _stop_response("agent.py is the Claude agent.")
    responses = [coding_response, stop]
    call_idx = [0]

    def fake_post(url, json=None, **kwargs):
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    captured = {}

    def capturing_execute_tool(name, args, shell, web, code, macos, guardrails, **kwargs):
        captured["coding"] = kwargs.get("coding")
        return "agent.py is the Claude agent."

    with patch.object(agent._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", side_effect=capturing_execute_tool):
            agent.run("What is agent.py?", cwd="/p")

    assert captured.get("coding") is agent._coding


# ── MCP dynamic tool injection ────────────────────────────────────────────────

def test_ollama_agent_includes_mcp_tools_in_tool_list():
    from tools.mcp import MCPManager
    a = OllamaAgent({"anthropic_api_key": "x", "brave_api_key": None}, Guardrails({}))
    mgr = MCPManager([{"name": "fs", "command": "npx", "args": [], "transport": "stdio"}])
    t = MagicMock()
    t.name = "read_file"
    t.description = "Read"
    t.inputSchema = {"type": "object", "properties": {}}
    mgr._inject_tools("fs", [t])
    a._mcp_manager = mgr
    tools = a._build_tool_list()
    names = [tool["function"]["name"] for tool in tools]
    assert "mcp__fs__read_file" in names
    assert "shell_run" in names


def test_ollama_agent_build_tool_list_without_mcp_returns_only_builtins():
    a = OllamaAgent({"anthropic_api_key": "x", "brave_api_key": None}, Guardrails({}))
    a._mcp_manager = None
    tools = a._build_tool_list()
    names = [tool["function"]["name"] for tool in tools]
    assert "shell_run" in names
    assert not any(n.startswith("mcp__") for n in names)


def test_ollama_agent_passes_mcp_manager_to_execute_tool():
    """OllamaAgent must pass mcp_manager when dispatching MCP tool calls."""
    from tools.mcp import MCPManager
    import tools._dispatch as dispatch_mod

    a = OllamaAgent({"anthropic_api_key": "x", "brave_api_key": None}, Guardrails({"guardrails": {"mcp_tool": "auto_allow"}}))
    mgr = MCPManager([{"name": "gh", "command": "x", "args": [], "transport": "stdio"}])
    t = MagicMock(); t.name = "list_issues"; t.description = ""; t.inputSchema = {"type": "object", "properties": {}}
    mgr._inject_tools("gh", [t])
    a._mcp_manager = mgr

    tool_resp = _tool_response("mcp__gh__list_issues", {})
    stop_resp = _stop_response("Here are the issues.")

    responses = [tool_resp, stop_resp]
    call_idx = [0]
    def fake_post(url, json=None, **kwargs):
        r = responses[call_idx[0]]; call_idx[0] += 1; return r

    captured = {}
    def spy_execute(*args, **kwargs):
        captured["mcp_manager"] = kwargs.get("mcp_manager")
        return "[]"

    with patch.object(a._http_client, "post", side_effect=fake_post):
        with patch("ollama_agent.execute_tool", side_effect=spy_execute):
            a.run("list issues")

    assert captured.get("mcp_manager") is mgr


# ── Task 12: result_summary cap and wrap-up nudge ─────────────────────────────

def test_ollama_result_summary_capped_at_200_chars(agent):
    long_result = "y" * 300
    tool_resp = _tool_response("shell_run", {"command": "ls"})
    stop_resp = _stop_response("Done.")

    responses = [tool_resp, stop_resp]
    idx = [0]
    def fake_post(url, json=None, **kwargs):
        r = responses[idx[0]]; idx[0] += 1; return r

    stream_mock = MagicMock(side_effect=ValueError("force fallback to post"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=fake_post):
            with patch("ollama_agent.execute_tool", return_value=long_result):
                result = agent.run("list files")

    assert len(result["steps"][0]["result_summary"]) <= 200


def test_ollama_wrap_up_nudge_injected_near_step_limit(agent):
    agent._config["reasoning"]["max_steps_ollama"] = 5
    agent._config["reasoning"]["stall_detection"] = False  # avoid near-dup terminating early
    call_payloads = []
    # Alternate tool names to also avoid near-duplicate detection
    tools_cycle = ["shell_run", "file_read", "shell_run", "file_read"]

    def fake_post(url, json=None, **kwargs):
        call_payloads.append(json)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        n = len(call_payloads)
        if n >= 4:
            resp.json.return_value = {"choices": [{"finish_reason": "stop",
                                                    "message": {"content": "Done.", "tool_calls": None}}]}
        else:
            tool_name = tools_cycle[(n - 1) % len(tools_cycle)]
            args = f'{{"command": "ls /tmp/{n}"}}' if tool_name == "shell_run" else f'{{"path": "/tmp/{n}.txt"}}'
            resp.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": f"c{n}", "function": {"name": tool_name, "arguments": args}}]
            }}]}
        return resp

    stream_mock = MagicMock(side_effect=ValueError("force fallback to post"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=fake_post):
            with patch("ollama_agent.execute_tool", return_value="ok"):
                agent.run("do stuff")

    assert len(call_payloads) >= 4
    nudge_found = any(
        "approaching your step limit" in (m.get("content") or "")
        for m in call_payloads[3]["messages"]
    )
    assert nudge_found, "wrap-up nudge not found in 4th API call's messages"


# ── planning text nudge ───────────────────────────────────────────────────────

def test_planning_text_triggers_nudge_then_tool_call(agent):
    """Model returns 'Let me check...' first, then a real tool call after nudge."""
    from ollama_agent import _is_planning_text
    assert _is_planning_text("Let me fetch the PR diff.")
    assert _is_planning_text("I'll check the CI failures for you.")
    assert not _is_planning_text("Done.")
    assert not _is_planning_text("Your Downloads folder is empty.")

    responses = [
        _stop_response("Let me check the CI failures for you."),  # planning text, no tools
        _tool_response("shell_run", {"command": "gh pr view 38 --json statusCheckRollup"}),
        _stop_response("CI failed due to missing mcp module."),
    ]
    idx = [0]
    captured_nudge = [False]

    def fake_post(url, json=None, **kwargs):
        r = responses[idx[0]]; idx[0] += 1; return r

    def fake_step(event):
        if event.get("type") == "clear":
            captured_nudge[0] = True

    stream_mock = MagicMock(side_effect=ValueError("force fallback"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=fake_post):
            with patch("ollama_agent.execute_tool", return_value="ok"):
                result = agent.run("check CI", step_callback=fake_step)

    assert captured_nudge[0], "clear SSE event not emitted on nudge"
    assert result["speak"] == "CI failed due to missing mcp module."
    assert len(result["steps"]) >= 1


def test_planning_text_escalates_after_2_retries(agent):
    """If the model returns planning text 3 times (0 tool calls), escalate to cloud."""
    stream_mock = MagicMock(side_effect=ValueError("force fallback"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post",
                          return_value=_stop_response("Let me check that for you.")):
            from ollama_agent import EscalateToCloud
            with pytest.raises(EscalateToCloud):
                agent.run("check something")


def _make_streaming_finalize_response(answer: str, call_id: str = "call_fin"):
    """Build SSE lines simulating a streaming finalize(answer=...) tool call."""
    args_str = _json_mod.dumps({"answer": answer})
    mid = len(args_str) // 2
    return [
        f'data: {_json_mod.dumps({"choices": [{"delta": {"role": "assistant", "content": "", "tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": "finalize", "arguments": ""}}]}, "finish_reason": None}]})}',
        f'data: {_json_mod.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_str[:mid]}}]}, "finish_reason": None}]})}',
        f'data: {_json_mod.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_str[mid:]}}]}, "finish_reason": None}]})}',
        f'data: {_json_mod.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})}',
        "data: [DONE]",
    ]


def test_finalize_streams_answer_as_tokens(agent):
    """When Ollama calls finalize(answer=...), the answer must be streamed as token events."""
    from ollama_agent import _stream_call
    answer = "Here is your code review: looks great overall."
    lines = _make_streaming_finalize_response(answer)

    events = []
    with patch.object(agent._http_client, "stream", return_value=_mock_stream(lines)):
        _stream_call(agent._http_client, "http://localhost/v1/chat/completions",
                     {"model": "test", "messages": [], "tools": []},
                     step_callback=lambda e: events.append(e))

    token_events = [e for e in events if e.get("type") == "token"]
    assert len(token_events) > 0, "Expected token events from finalize answer"
    assembled = "".join(e["text"] for e in token_events)
    assert answer in assembled or assembled in answer, (
        f"Assembled tokens '{assembled}' do not match expected answer '{answer}'"
    )


def test_finalize_emits_composing_step_before_tokens(agent):
    """A 'Composing response…' step event must fire before the first token of a finalize answer."""
    from ollama_agent import _stream_call
    lines = _make_streaming_finalize_response("The answer is 42.")

    events = []
    composing_state = [False]
    with patch.object(agent._http_client, "stream", return_value=_mock_stream(lines)):
        _stream_call(agent._http_client, "http://localhost/v1/chat/completions",
                     {"model": "test", "messages": [], "tools": []},
                     step_callback=lambda e: events.append(e),
                     composing_state=composing_state)

    step_events = [e for e in events if e.get("type") == "step"]
    composing = [e for e in step_events if e.get("label") == "Composing response…"]
    assert composing, "Expected 'Composing response…' step event for finalize call"

    first_token_idx = next((i for i, e in enumerate(events) if e.get("type") == "token"), len(events))
    composing_idx = next(i for i, e in enumerate(events) if e.get("label") == "Composing response…")
    assert composing_idx < first_token_idx, "Composing step must fire before first token"


def test_planning_text_after_tool_calls_triggers_nudge(agent):
    """Model makes some tool calls, then returns planning text — should get nudged, not exit."""
    responses = [
        _tool_response("mcp__github__list_pull_requests", {"owner": "matanvilensky", "repo": "jarvis"}),
        # After the tool call, model returns planning text instead of next tool or finalize
        _stop_response("Let me check for PRs in the repo and review the local branch changes."),
        # After nudge, model calls the corrected tool
        _tool_response("mcp__github__list_pull_requests", {"owner": "Matanvil", "repo": "jarvis"}),
        _stop_response("PR #39 is open: feat: Phase E."),
    ]
    idx = [0]
    nudge_fired = [False]

    def fake_post(url, json=None, **kwargs):
        r = responses[idx[0]]; idx[0] += 1; return r

    def fake_step(event):
        if event.get("type") == "clear":
            nudge_fired[0] = True

    stream_mock = MagicMock(side_effect=ValueError("force fallback"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=fake_post):
            with patch("ollama_agent.execute_tool", return_value='[{"number":39}]'):
                result = agent.run("check PRs", step_callback=fake_step)

    assert nudge_fired[0], "clear SSE not emitted when planning text followed tool calls"
    assert "PR #39" in result["speak"]


# ── action trace detection ───────────────────────────────────────────────────

def test_action_trace_response_triggers_nudge_and_retry(agent):
    """Model returning 'Actions: ...' text must be nudged to call finalize() instead."""
    action_trace = (
        "Actions:\n"
        "- shell_run({'command': 'date'}) → exit_code=0\nstdout=Wednesday\n\nResult: Today is Wednesday."
    )
    responses = [
        _stop_response(action_trace),
        _stop_response("Today is Wednesday, June 18th."),
    ]
    idx = [0]
    stream_mock = MagicMock(side_effect=ValueError("force fallback to post"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=lambda url, json=None, **kw: (idx.__setitem__(0, idx[0]+1) or responses[idx[0]-1])):
            result = agent.run("what day is it today?")
    assert "Wednesday" in result["speak"]
    assert not result["speak"].startswith("Actions:"), "action trace must not leak into final answer"


def test_action_trace_does_not_enter_history_as_valid_answer(agent):
    """The 'Actions:' response should be appended to messages as assistant content
    (for context) but the next user message should be a nudge, not a finalize return."""
    action_trace = "Actions:\n- shell_run({'command': 'date'}) → exit_code=0\nstdout=Mon\n\nResult: Monday."
    calls = []
    def fake_post(url, json=None, **kw):
        calls.append(json)
        if len(calls) == 1:
            return _stop_response(action_trace)
        return _stop_response("It's Monday.")

    stream_mock = MagicMock(side_effect=ValueError("force fallback"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=fake_post):
            result = agent.run("what day is it?")

    # Second call's messages should contain a nudge about the Actions: format
    assert len(calls) >= 2
    messages_in_second_call = calls[1].get("messages", [])
    user_nudge = next(
        (m for m in messages_in_second_call if m["role"] == "user" and "finalize" in m["content"].lower()),
        None
    )
    assert user_nudge is not None, "nudge message about finalize() must be sent after action trace"
    assert "It's Monday" in result["speak"]


# ── enable_thinking retry ─────────────────────────────────────────────────────

def _empty_length_response() -> MagicMock:
    """Simulate model returning empty content with finish_reason=length (token budget exhausted)."""
    msg = MagicMock()
    msg.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": None},
                     "finish_reason": "length"}]
    }
    return msg


def test_thinking_exhausted_retries_without_thinking(agent):
    """When complex_reasoning + enable_thinking exhausts token budget, retry with thinking off."""
    payloads_sent = []

    def fake_post(url, json=None, **kwargs):
        payloads_sent.append(json or {})
        # First call (thinking on) → empty/length; second call (thinking off) → real answer
        if json and json.get("enable_thinking", False):
            return _empty_length_response()
        return _stop_response("Noam Shazeer invented the Transformer attention mechanism.")

    stream_mock = MagicMock(side_effect=ValueError("force fallback to post"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=fake_post):
            result = agent.run("tell me about Noam Shazeer", intent_class="complex_reasoning")

    assert result["speak"], "Expected a non-empty answer after thinking retry"
    assert "Shazeer" in result["speak"] or "Transformer" in result["speak"]

    thinking_on_calls = [p for p in payloads_sent if p.get("enable_thinking") is True]
    thinking_off_calls = [p for p in payloads_sent if p.get("enable_thinking") is False]
    assert len(thinking_on_calls) >= 1, "Expected at least one thinking=True attempt"
    assert len(thinking_off_calls) >= 1, "Expected fallback attempt with thinking=False"


def test_thinking_not_used_for_non_complex_intents(agent):
    """enable_thinking must be False for non-complex_reasoning intent classes."""
    payloads_sent = []

    def fake_post(url, json=None, **kwargs):
        payloads_sent.append(json or {})
        return _stop_response("The answer is 42.")

    stream_mock = MagicMock(side_effect=ValueError("force fallback to post"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", side_effect=fake_post):
            agent.run("what is 6 times 7", intent_class="read_only")

    for p in payloads_sent:
        assert p.get("enable_thinking") is False, "enable_thinking must be False for read_only intent"


def test_thinking_escalates_if_retry_also_empty(agent):
    """If both thinking=True and thinking=False return empty, escalate to cloud."""
    stream_mock = MagicMock(side_effect=ValueError("force fallback to post"))
    with patch.object(agent._http_client, "stream", stream_mock):
        with patch.object(agent._http_client, "post", return_value=_empty_length_response()):
            with pytest.raises(EscalateToCloud):
                agent.run("something complex", intent_class="complex_reasoning")
