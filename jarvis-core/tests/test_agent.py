import pytest
from unittest.mock import patch, MagicMock
from agent import Agent, claude_code_available, _step_label
from guardrails import Guardrails, Decision
from tools._dispatch import execute_tool, format_response


def make_agent(model: str = "claude-haiku-4-5-20251001"):
    config = {
        "anthropic_api_key": "sk-test",
        "guardrails": {
            "run_shell": "auto_allow",
            "run_code_with_effects": "auto_allow",
            "read_files": "auto_allow",
            "edit_files": "auto_allow",
            "web_search": "auto_allow",
            "open_apps": "auto_allow",
            "delete_files": "require_approval",
        },
    }
    guardrails = Guardrails(config)
    return Agent(config=config, guardrails=guardrails, model=model)


def test_agent_uses_specified_model():
    agent = make_agent(model="claude-sonnet-4-6")
    assert agent._model == "claude-sonnet-4-6"


def test_agent_default_model_is_haiku():
    agent = make_agent()
    assert agent._model == "claude-haiku-4-5-20251001"


def test_agent_passes_model_to_api_call():
    agent = make_agent(model="claude-haiku-4-5-20251001")
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(text="Done.", spec=["text"])]
    with patch.object(agent._client.messages, "create", return_value=mock_response) as mock_create:
        agent.run("hello")
    assert mock_create.call_args.kwargs["model"] == "claude-haiku-4-5-20251001"


def test_agent_formats_short_response_for_voice():
    result = format_response("Done!", tool_calls_made=[])
    assert result["speak"] == "Done!"
    assert result["display"] == "Done!"


def test_agent_formats_long_response_speak_summary():
    long_text = "x" * 200
    result = format_response(long_text, tool_calls_made=["shell_run"])
    assert len(result["speak"]) < len(long_text)
    assert result["display"] == long_text


def test_agent_formats_code_response():
    code_response = "Here is the code:\n```python\ndef hello():\n    pass\n```"
    result = format_response(code_response, tool_calls_made=["run_code"])
    assert "```" not in result["speak"]
    assert "```" in result["display"]


def test_format_response_voice_tag_used_as_speak():
    text = "I searched the web and found 5 results about Python.\nVOICE: Found 5 results about Python."
    result = format_response(text, tool_calls_made=["web_search"])
    assert result["speak"] == "Found 5 results about Python."
    assert "VOICE:" not in result["display"]
    assert "I searched the web" in result["display"]


def test_format_response_voice_tag_stripped_from_display():
    text = "Created file hello.py with the requested content.\nVOICE: Done, created hello.py."
    result = format_response(text, tool_calls_made=["file_write"])
    assert "VOICE:" not in result["display"]
    assert result["speak"] == "Done, created hello.py."


def test_format_response_no_voice_tag_falls_back_to_heuristic():
    long_text = "x" * 200   # no sentences, no VOICE tag → truncation heuristic
    result = format_response(long_text, tool_calls_made=[])
    assert len(result["speak"]) <= 141   # heuristic caps at 140 + "…"
    assert result["display"] == long_text


def test_web_fetch_empty_body_returns_empty_string_not_error():
    """web_fetch with empty string body should return '' not 'error=None'."""
    agent = make_agent()
    with patch.object(agent._web, "fetch_page", return_value={"text": "", "error": None}):
        result = execute_tool(
            "web_fetch", {"url": "https://example.com"},
            agent._shell, agent._web, agent._code, agent._macos, agent._guardrails,
            default_cwd=None,
        )
    assert result == ""


def test_guardrails_block_is_surfaced():
    agent = make_agent()
    from guardrails import Action
    action = Action(category="delete_files", description="rm ~/important.txt")
    decision = agent._guardrails.classify(action)
    assert decision == Decision.REQUIRE_APPROVAL


def test_execute_tool_shell_run_uses_cwd():
    agent = make_agent()
    with patch.object(agent._shell, "run", return_value={"exit_code": 0, "stdout": "ok", "stderr": "", "error": None}) as mock_run:
        execute_tool("shell_run", {"command": "pytest"}, agent._shell, agent._web, agent._code, agent._macos, agent._guardrails, default_cwd="/my/project")
    mock_run.assert_called_once_with("pytest", cwd="/my/project")


def test_execute_tool_cwd_override_from_input():
    agent = make_agent()
    with patch.object(agent._shell, "run", return_value={"exit_code": 0, "stdout": "", "stderr": "", "error": None}) as mock_run:
        execute_tool("shell_run", {"command": "ls", "cwd": "/override"}, agent._shell, agent._web, agent._code, agent._macos, agent._guardrails, default_cwd="/default")
    mock_run.assert_called_once_with("ls", cwd="/override")


def test_execute_tool_run_code_multi_language():
    agent = make_agent()
    with patch.object(agent._code, "run_snippet", return_value={"exit_code": 0, "stdout": "hi", "stderr": "", "error": None}) as mock_run:
        execute_tool("run_code", {"code": 'console.log("hi")', "language": "javascript"}, agent._shell, agent._web, agent._code, agent._macos, agent._guardrails, default_cwd=None)
    mock_run.assert_called_once_with('console.log("hi")', "javascript", cwd=None)


def test_approval_required_returns_structured_response():
    agent = make_agent()
    mock_block = MagicMock()
    mock_block.name = "shell_run"
    mock_block.input = {"command": "rm -rf /important"}
    mock_block.id = "tool_123"
    mock_block.type = "tool_use"

    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = [mock_block]

    # delete_files requires approval — but shell_run maps to run_shell which is auto_allow
    # so let's use a tool mapped to a blocked category
    with patch.object(agent._guardrails, "classify", return_value=Decision.REQUIRE_APPROVAL):
        with patch.object(agent._client.messages, "create", return_value=mock_response):
            result = agent.run("delete my downloads folder")

    assert "approval_required" in result
    assert result["approval_required"]["tool"] == "shell_run"


def test_delegate_to_claude_code_runs_cli(tmp_path):
    agent = make_agent()
    with patch("tools._dispatch._claude_code_available", return_value=True), \
         patch.object(agent._shell, "run", return_value={"exit_code": 0, "stdout": "All done.", "stderr": "", "error": None}) as mock_run:
        result = execute_tool(
            "delegate_to_claude_code",
            {"task": "review PR #42 and apply suggested changes"},
            agent._shell, agent._web, agent._code, agent._macos, agent._guardrails,
            default_cwd=str(tmp_path),
        )
    assert result == "All done."
    call_kwargs = mock_run.call_args
    assert str(tmp_path) == call_kwargs[1]["cwd"]
    assert "claude" in call_kwargs[0][0]


def test_delegate_to_claude_code_missing_cli():
    agent = make_agent()
    with patch("tools._dispatch._claude_code_available", return_value=False):
        result = execute_tool(
            "delegate_to_claude_code",
            {"task": "do something"},
            agent._shell, agent._web, agent._code, agent._macos, agent._guardrails,
            default_cwd=None,
        )
    assert "not installed" in result


def test_claude_code_available_reflects_system():
    # Just verify it returns a bool without crashing
    result = claude_code_available()
    assert isinstance(result, bool)


def test_run_injects_memory_context_into_system_prompt(tmp_path, monkeypatch):
    """If memory_context is provided, it appears in the system prompt."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    cfg = config.load()
    from guardrails import Guardrails
    from agent import Agent
    guardrails = Guardrails(cfg)
    agent = Agent(config=cfg, guardrails=guardrails)

    captured_prompts = []

    def fake_create(**kwargs):
        captured_prompts.append(kwargs.get("system", ""))
        resp = MagicMock()
        resp.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.text = "Done."
        resp.content = [text_block]
        return resp

    with patch.object(agent._client.messages, "create", side_effect=fake_create):
        agent.run("run tests", cwd="/my/project", memory_context="Test: npm test | Build: npm run build")

    assert len(captured_prompts) == 1
    assert "npm test" in captured_prompts[0]


def test_delegate_to_local_calls_ollama_agent(tmp_path, monkeypatch):
    """When Claude calls delegate_to_local, OllamaAgent.run() is called."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    from guardrails import Guardrails
    from agent import Agent
    cfg = config.load()
    guardrails = Guardrails(cfg)

    mock_local = MagicMock()
    mock_local.run.return_value = {"speak": "Found 3 files.", "display": "Found 3 files."}
    agent = Agent(config=cfg, guardrails=guardrails, local_agent=mock_local)

    # Simulate Claude calling delegate_to_local
    from tools._dispatch import execute_tool
    result = execute_tool(
        "delegate_to_local",
        {"task": "find all .tsx files"},
        agent._shell, agent._web, agent._code, agent._macos,
        guardrails,
        local_agent=mock_local,
    )
    mock_local.run.assert_called_once_with("find all .tsx files", cwd=None)
    assert "Found 3 files" in result


def test_stall_detection_injects_warning(tmp_path, monkeypatch):
    """If the same tool is called with the same input twice, inject a stall warning."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    from guardrails import Guardrails
    from agent import Agent
    cfg = config.load()
    guardrails = Guardrails(cfg)
    agent = Agent(config=cfg, guardrails=guardrails)

    call_log = []

    def fake_create(**kwargs):
        call_log.append(kwargs.get("messages", []))
        resp = MagicMock()
        if len(call_log) < 3:
            # First 2 calls: always call shell_run with same args (stall)
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = "shell_run"
            tool_block.input = {"command": "ls /tmp"}
            tool_block.id = f"call_{len(call_log)}"
            resp.stop_reason = "tool_use"
            resp.content = [tool_block]
        else:
            resp.stop_reason = "end_turn"
            text_block = MagicMock()
            text_block.text = "Done."
            resp.content = [text_block]
        return resp

    with patch.object(agent._client.messages, "create", side_effect=fake_create), \
         patch("tools._dispatch.execute_tool", return_value="exit_code=0\nstdout=tmp\nstderr="):
        agent.run("do something")

    # Check that a stall warning was injected somewhere in the messages
    all_messages = [msg for call in call_log for msg in call]
    stall_msg = next(
        (m for m in all_messages
         if isinstance(m.get("content"), str) and "already tried" in m["content"].lower()),
        None
    )
    assert stall_msg is not None


def test_run_returns_steps_list(tmp_path, monkeypatch):
    """Response includes a steps list with tool calls made."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    from guardrails import Guardrails
    from agent import Agent
    cfg = config.load()
    guardrails = Guardrails(cfg)
    agent = Agent(config=cfg, guardrails=guardrails)

    tool_resp = MagicMock()
    tool_resp.stop_reason = "tool_use"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "shell_run"
    tool_block.input = {"command": "ls"}
    tool_block.id = "call_1"
    tool_resp.content = [tool_block]

    end_resp = MagicMock()
    end_resp.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.text = "Done."
    end_resp.content = [text_block]

    with patch.object(agent._client.messages, "create", side_effect=[tool_resp, end_resp]), \
         patch("tools._dispatch.execute_tool", return_value="exit_code=0\nstdout=\nstderr="):
        result = agent.run("list files")

    assert "steps" in result
    assert len(result["steps"]) >= 1
    assert result["steps"][0]["tool"] == "shell_run"
    assert "milestone" in result["steps"][0]


def test_max_tokens_stop_reason_breaks_loop_gracefully(tmp_path, monkeypatch):
    """When Claude hits max_tokens, the loop should stop and return partial result."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    from guardrails import Guardrails
    from agent import Agent
    cfg = config.load()
    guardrails = Guardrails(cfg)
    agent = Agent(config=cfg, guardrails=guardrails)

    truncated_resp = MagicMock()
    truncated_resp.stop_reason = "max_tokens"
    text_block = MagicMock()
    text_block.text = "Partial answer..."
    truncated_resp.content = [text_block]

    call_count = []

    def fake_create(**kwargs):
        call_count.append(1)
        return truncated_resp

    with patch.object(agent._client.messages, "create", side_effect=fake_create):
        result = agent.run("write a novel")

    # Should stop after first max_tokens response, not burn through all max_steps
    assert len(call_count) == 1
    assert "speak" in result
    assert "steps" in result


def test_first_tool_call_is_milestone(tmp_path, monkeypatch):
    """The first tool call in a run is always a milestone."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config
    from guardrails import Guardrails
    from agent import Agent
    cfg = config.load()
    guardrails = Guardrails(cfg)
    agent = Agent(config=cfg, guardrails=guardrails)

    tool_resp = MagicMock()
    tool_resp.stop_reason = "tool_use"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "file_read"
    tool_block.input = {"path": "/tmp/test.txt"}
    tool_block.id = "call_1"
    tool_resp.content = [tool_block]

    end_resp = MagicMock()
    end_resp.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.text = "Done."
    end_resp.content = [text_block]

    with patch.object(agent._client.messages, "create", side_effect=[tool_resp, end_resp]), \
         patch("tools._dispatch.execute_tool", return_value="content"):
        result = agent.run("read test file")

    assert result["steps"][0]["milestone"] is True


# --- Schedule tool tests ---

@pytest.fixture
def haiku_agent():
    return make_agent()


def test_handle_schedule_tool_create(haiku_agent):
    from schedule_store import Schedule
    from datetime import datetime, timezone
    import scheduler as sched_module

    sample = Schedule(
        id="abc123", label="morning summary", command="summarise my calendar",
        schedule_type="recurring", cron="0 9 * * *", run_at_iso=None,
        enabled=True, created_at=datetime.now(timezone.utc).isoformat(), output="telegram",
    )
    mock_sched = MagicMock()
    mock_sched.create.return_value = sample

    with patch.object(sched_module, "_scheduler", mock_sched):
        from agent import _handle_schedule_tool
        result = _handle_schedule_tool("create_schedule", {
            "command": "summarise my calendar",
            "label": "morning summary",
            "schedule_type": "recurring",
            "cron": "0 9 * * *",
            "run_at_iso": None,
        })
    assert result["id"] == "abc123"
    assert result["label"] == "morning summary"


def test_handle_schedule_tool_list(haiku_agent):
    import scheduler as sched_module
    mock_sched = MagicMock()
    mock_sched.list.return_value = []
    with patch.object(sched_module, "_scheduler", mock_sched):
        from agent import _handle_schedule_tool
        result = _handle_schedule_tool("list_schedules", {})
    assert result == {"schedules": []}


def test_handle_schedule_tool_delete(haiku_agent):
    import scheduler as sched_module
    mock_sched = MagicMock()
    mock_sched.delete.return_value = True
    with patch.object(sched_module, "_scheduler", mock_sched):
        from agent import _handle_schedule_tool
        result = _handle_schedule_tool("delete_schedule", {"schedule_id": "abc123"})
    assert result["ok"] is True


def test_handle_schedule_tool_pause(haiku_agent):
    from schedule_store import Schedule
    from datetime import datetime, timezone
    import scheduler as sched_module

    sample = Schedule(
        id="abc123", label="l", command="c", schedule_type="recurring", cron="0 9 * * *",
        run_at_iso=None, enabled=False, created_at=datetime.now(timezone.utc).isoformat(),
        output="telegram",
    )
    mock_sched = MagicMock()
    mock_sched.pause.return_value = sample
    with patch.object(sched_module, "_scheduler", mock_sched):
        from agent import _handle_schedule_tool
        result = _handle_schedule_tool("pause_schedule", {"schedule_id": "abc123"})
    assert result["enabled"] is False


def test_handle_schedule_tool_resume(haiku_agent):
    from schedule_store import Schedule
    from datetime import datetime, timezone
    import scheduler as sched_module

    sample = Schedule(
        id="abc123", label="l", command="c", schedule_type="recurring", cron="0 9 * * *",
        run_at_iso=None, enabled=True, created_at=datetime.now(timezone.utc).isoformat(),
        output="telegram",
    )
    mock_sched = MagicMock()
    mock_sched.resume.return_value = sample
    with patch.object(sched_module, "_scheduler", mock_sched):
        from agent import _handle_schedule_tool
        result = _handle_schedule_tool("resume_schedule", {"schedule_id": "abc123"})
    assert result["enabled"] is True


def test_handle_schedule_tool_no_scheduler():
    import scheduler as sched_module
    with patch.object(sched_module, "_scheduler", None):
        from agent import _handle_schedule_tool
        result = _handle_schedule_tool("list_schedules", {})
    assert "error" in result


def test_step_label_known_tools():
    assert _step_label("shell_run") == "Running command"
    assert _step_label("file_read") == "Reading file"
    assert _step_label("file_edit") == "Editing file"
    assert _step_label("file_write") == "Editing file"
    assert _step_label("web_search") == "Searching the web"
    assert _step_label("delegate_to_local") == "Thinking locally"
    assert _step_label("delegate_to_claude_code") == "Delegating to Claude Code"
    assert _step_label("create_schedule") == "Creating schedule"
    assert _step_label("pause_schedule") == "Pausing schedule"
    assert _step_label("resume_schedule") == "Resuming schedule"
    assert _step_label("search_content") == "Searching content"

def test_step_label_unknown_tool():
    assert _step_label("some_future_tool") == "Working\u2026"


def test_step_callback_called_at_milestone():
    """step_callback is called for milestone steps with correct payload."""
    called = []
    def cb(event):
        called.append(event)

    agent = make_agent()
    # Patch _client.messages.create to return one tool_use block then end_turn
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tu_1"
    tool_block.name = "shell_run"
    tool_block.input = {"command": "echo hi"}

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Done."

    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.content = [tool_block]

    resp2 = MagicMock()
    resp2.stop_reason = "end_turn"
    resp2.content = [text_block]

    with patch.object(agent._client.messages, "create", side_effect=[resp1, resp2]):
        with patch("agent.execute_tool", return_value="output"):
            agent.run("do something", step_callback=cb)

    assert len(called) == 1
    assert called[0]["type"] == "step"
    assert called[0]["label"] == "Running command"
    assert called[0]["tool"] == "shell_run"
    assert called[0]["milestone"] is True


def test_step_callback_not_called_for_non_milestone():
    """step_callback is NOT called for non-milestone steps."""
    called = []

    agent = make_agent()
    # First tool (milestone=True), second tool (milestone=False for file_read at index 1)
    block1 = MagicMock()
    block1.type = "tool_use"
    block1.id = "tu_1"
    block1.name = "shell_run"
    block1.input = {"command": "ls"}

    block2 = MagicMock()
    block2.type = "tool_use"
    block2.id = "tu_2"
    block2.name = "file_read"
    block2.input = {"path": "/tmp/f"}

    text = MagicMock()
    text.type = "text"
    text.text = "Done."

    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.content = [block1]

    resp2 = MagicMock()
    resp2.stop_reason = "tool_use"
    resp2.content = [block2]

    resp3 = MagicMock()
    resp3.stop_reason = "end_turn"
    resp3.content = [text]

    with patch.object(agent._client.messages, "create", side_effect=[resp1, resp2, resp3]):
        with patch("agent.execute_tool", return_value="output"):
            agent.run("do something", step_callback=lambda e: called.append(e))

    # Only first step is milestone
    assert len(called) == 1
    assert called[0]["tool"] == "shell_run"
