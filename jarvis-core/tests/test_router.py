import os
import pytest
from datetime import date
from unittest.mock import MagicMock, patch
from guardrails import Guardrails
from local_agent import EscalateToCloud
from router import Router
from prompt_loader import PromptLoader


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config as cfg_module
    return cfg_module.load()


@pytest.fixture
def mock_local_agent():
    agent = MagicMock()
    agent.run.return_value = {"speak": "Done locally.", "display": "Done locally.", "error": None}
    return agent


@pytest.fixture
def mock_claude_agent():
    agent = MagicMock()
    agent.run.return_value = {"speak": "Done via Claude.", "display": "Done via Claude.", "error": None}
    return agent


@pytest.fixture
def router(config, mock_local_agent, mock_claude_agent):
    # Default fixture: automatic mode
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._local = mock_local_agent
    r._claude = mock_claude_agent
    r._sonnet = mock_claude_agent
    # Pre-flight classifier is mocked by default; individual tests can override
    r._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "read_only", "reason": "test"})
    return r


# ── routing_mode: automatic ───────────────────────────────────────────────────

def test_automatic_uses_local_when_successful(router, mock_local_agent, mock_claude_agent):
    result = router.process("list Downloads", cwd=None)
    called_kwargs = mock_local_agent.run.call_args.kwargs
    assert called_kwargs["cwd"] is None
    assert "memory_context" not in called_kwargs  # now baked into user_text prefix
    assert isinstance(called_kwargs["history"], list)
    mock_claude_agent.run.assert_not_called()
    assert result["speak"] == "Done locally."


def test_automatic_falls_back_to_sonnet_when_local_escalates(router, mock_local_agent, mock_claude_agent):
    """When local raises EscalateToCloud in automatic mode, router falls through to Sonnet."""
    mock_local_agent.run.side_effect = EscalateToCloud("connection refused")
    result = router.process("open Safari", cwd=None)
    mock_claude_agent.run.assert_called_once()
    assert result["_agent"] == "claude"
    assert result["_escalated"] is True


# ── routing_mode: cloud ───────────────────────────────────────────────────────

def test_cloud_mode_skips_local(router, mock_local_agent, mock_claude_agent, config):
    config["local"]["routing_mode"] = "cloud"
    result = router.process("any command", cwd=None)
    mock_local_agent.run.assert_not_called()
    mock_claude_agent.run.assert_called_once()
    assert result["speak"] == "Done via Claude."


# ── routing_mode: local ───────────────────────────────────────────────────────

def test_local_mode_never_calls_cloud(router, mock_local_agent, mock_claude_agent, config):
    config["local"]["routing_mode"] = "local"
    mock_local_agent.run.return_value = {"speak": "Best I can do offline.", "display": "Best I can do offline."}
    result = router.process("search the web", cwd=None)
    mock_claude_agent.run.assert_not_called()
    assert result["speak"] == "Best I can do offline."


# ── analytics logging ─────────────────────────────────────────────────────────

def test_process_returns_agent_metadata(router):
    result = router.process("list Downloads", cwd=None)
    assert result.get("_agent") == "local"
    assert result.get("_escalated") is False
    assert isinstance(result.get("_response_ms"), int)


def test_process_returns_intent_class_in_metadata(router):
    """Intent class from pre-flight classifier should appear in response metadata."""
    result = router.process("list Downloads", cwd=None)
    assert result.get("_agent") == "local"
    assert result.get("_escalated") is False
    assert result.get("_intent_class") == "read_only"


def test_local_mode_guard_records_escalation_truthfully(router, mock_local_agent, mock_claude_agent, config):
    """If LocalAgent leaks EscalateToCloud in local mode, router must not call cloud
    and must record escalated=True with the suppression reason."""
    config["local"]["routing_mode"] = "local"
    mock_local_agent.run.side_effect = EscalateToCloud("unexpected reason")
    result = router.process("something", cwd=None)
    mock_claude_agent.run.assert_not_called()
    assert result.get("_escalated") is True
    assert "suppressed:local" in result.get("_escalation_reason", "")


def test_local_mode_with_unreachable_local_does_not_call_cloud(router, mock_local_agent, mock_claude_agent, config):
    """When local is unreachable in local mode, Router must not escalate to cloud."""
    config["local"]["routing_mode"] = "local"
    mock_local_agent.run.side_effect = EscalateToCloud("unavailable: connection refused")
    result = router.process("open Safari", cwd=None)
    mock_claude_agent.run.assert_not_called()
    assert result.get("_escalated") is True
    assert "suppressed:local" in result.get("_escalation_reason", "")
    assert result.get("speak") is not None  # returned something, not None


# ── routing_mode validation ───────────────────────────────────────────────────

def test_valid_modes_are_local_cloud_automatic():
    """Only local, cloud, automatic are valid routing modes."""
    from router import _VALID_ROUTING_MODES
    assert _VALID_ROUTING_MODES == {"local", "cloud", "automatic"}


def test_invalid_routing_mode_logs_warning(config):
    """Invalid routing_mode should log a warning, not crash."""
    import logging
    config["local"]["routing_mode"] = "invalid_mode"
    guardrails = Guardrails(config)
    # Should not raise — warning goes to named logger (jarvis.errors), not root logger
    r = Router(config=config, guardrails=guardrails)
    assert r is not None


def test_valid_routing_mode_does_not_log_warning(config):
    """Valid routing_mode should not log a warning."""
    config["local"]["routing_mode"] = "cloud"
    guardrails = Guardrails(config)
    with patch("logging.Logger.warning") as mock_warning:
        r = Router(config=config, guardrails=guardrails)
    mock_warning.assert_not_called()


# ── pre-flight classifier ──────────────────────────────────────────────────────

def test_classifier_routes_local_task_to_local(config):
    """read_only + can_handle_locally=True → LocalAgent, no cloud call."""
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_local = MagicMock()
    mock_cloud = MagicMock()
    mock_local.run.return_value = {"speak": "Done.", "display": "Done."}
    r._local = mock_local
    r._sonnet = mock_cloud

    classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "file op"}
    with patch.object(r, "_classify", return_value=classification):
        result = r.process("list my files", cwd=None)

    mock_local.run.assert_called_once()
    mock_cloud.run.assert_not_called()


def test_classifier_routes_complex_task_to_sonnet(config):
    """complex_reasoning → Sonnet, no local call."""
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_local = MagicMock()
    mock_sonnet = MagicMock()
    mock_sonnet.run.return_value = {"speak": "Done via Sonnet.", "display": "Done."}
    r._local = mock_local
    r._sonnet = mock_sonnet

    classification = {"can_handle_locally": False, "intent_class": "complex_reasoning", "reason": "web search"}
    with patch.object(r, "_classify", return_value=classification):
        result = r.process("latest React news", cwd=None)

    mock_local.run.assert_not_called()
    mock_sonnet.run.assert_called_once()
    assert result.get("_escalated") is False   # direct routing, not escalation


def test_classifier_failure_falls_back_to_local(config):
    """If _classify raises, Router falls back to local executor."""
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_local = MagicMock()
    mock_cloud = MagicMock()
    mock_local.run.return_value = {"speak": "Done.", "display": "Done."}
    r._local = mock_local
    r._sonnet = mock_cloud

    with patch.object(r, "_classify", side_effect=Exception("classifier down")):
        result = r.process("open Safari", cwd=None)

    mock_local.run.assert_called_once()   # fell back gracefully


# ── automatic mode fixtures and tests ─────────────────────────────────────────

@pytest.fixture
def mock_haiku_agent():
    agent = MagicMock()
    agent.run.return_value = {"speak": "Done via Haiku.", "display": "Done via Haiku.", "steps": []}
    return agent


@pytest.fixture
def mock_sonnet_agent():
    agent = MagicMock()
    agent.run.return_value = {"speak": "Done via Sonnet.", "display": "Done via Sonnet.", "steps": []}
    return agent


@pytest.fixture
def automatic_router(config, mock_local_agent, mock_haiku_agent, mock_sonnet_agent):
    config["local"]["routing_mode"] = "automatic"
    config["local"]["model"] = "qwen-executor"
    config["local"]["executor_model"] = "supergemma4-test"
    config["local"]["classifier_model"] = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._local = mock_local_agent
    r._haiku = mock_haiku_agent
    r._sonnet = mock_sonnet_agent
    r._claude = mock_sonnet_agent
    r._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "read_only", "reason": "test"})
    return r


def test_automatic_uses_local_for_simple_tasks(automatic_router, mock_local_agent, mock_sonnet_agent):
    result = automatic_router.process("list my files")
    mock_local_agent.run.assert_called_once()
    mock_sonnet_agent.run.assert_not_called()
    assert result["_agent"] == "local"
    assert result["_model"] == "supergemma4-test"


def test_automatic_uses_sonnet_for_complex_reasoning(automatic_router, mock_local_agent, mock_sonnet_agent):
    automatic_router._classify = MagicMock(return_value={
        "can_handle_locally": False, "intent_class": "complex_reasoning", "reason": "needs web"
    })
    result = automatic_router.process("search the web for latest news")
    mock_local_agent.run.assert_not_called()
    mock_sonnet_agent.run.assert_called_once()
    assert result["_agent"] == "claude"
    assert result["_model"] == "claude-sonnet-4-6"


def test_automatic_stays_local_on_escalate_suppressed_by_local_mode(config, mock_local_agent, mock_sonnet_agent):
    """In local mode, EscalateToCloud must return graceful message — never call cloud."""
    config["local"]["routing_mode"] = "local"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._local = mock_local_agent
    r._sonnet = mock_sonnet_agent
    r._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "read_only", "reason": "simple"
    })
    mock_local_agent.run.side_effect = EscalateToCloud("timeout")
    result = r.process("do something hard")
    mock_sonnet_agent.run.assert_not_called()
    assert result["_escalated"] is True
    assert "suppressed:local" in result["_escalation_reason"]
    assert result["speak"] is not None


def test_automatic_classifier_failure_falls_back_to_local(automatic_router, mock_local_agent, mock_sonnet_agent):
    automatic_router._classify = MagicMock(side_effect=Exception("classifier down"))
    result = automatic_router.process("hello")
    mock_local_agent.run.assert_called_once()
    mock_sonnet_agent.run.assert_not_called()


def test_classifier_uses_classifier_model(config):
    """_classifier_model should use classifier_model key when present."""
    config["local"]["model"] = "qwen-executor"
    config["local"]["classifier_model"] = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    assert r._classifier_model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert r._local_model == "qwen-executor"


def test_default_routing_mode_is_automatic(config):
    """Default routing mode should be automatic."""
    assert config["local"]["routing_mode"] == "automatic"


def test_default_routing_mode_in_defaults():
    """Default routing mode in DEFAULTS should be automatic."""
    from config import DEFAULTS
    assert DEFAULTS["local"]["routing_mode"] == "automatic"


def test_resume_returns_none_when_no_paused_run(automatic_router):
    import approval_store
    approval_store.clear()
    assert automatic_router.resume("nope") is None


def test_resume_invokes_stored_resumer_and_annotates(automatic_router):
    import approval_store
    approval_store.clear()
    approval_store.register(
        "c1",
        lambda step_callback=None: {"speak": "ok", "display": "ok", "steps": []},
        {"user_text": "hi", "agent": "local", "model": "m1"},
    )
    result = automatic_router.resume("c1")
    assert result["_agent"] == "local"
    assert result["_model"] == "m1"
    assert result["speak"] == "ok"
    assert not approval_store.has("c1")  # popped


def test_default_base_local_model_is_qwen36():
    """Default base local model should align with the selected Qwen executor."""
    from config import DEFAULTS
    assert DEFAULTS["local"]["model"] == "qwen3.6:35b-a3b"


def test_classifier_model_in_defaults():
    """classifier_model should be present in DEFAULTS and point to jarvis-classifier."""
    from config import DEFAULTS
    assert DEFAULTS["local"]["classifier_model"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def test_automatic_annotates_result_with_executor_model_name(automatic_router, mock_local_agent):
    """_model in result should reflect the local executor_model."""
    result = automatic_router.process("list my files")
    assert result["_model"] == "supergemma4-test"  # matches what automatic_router fixture sets


# ── history tool summary ──────────────────────────────────────────────────────

def test_history_does_not_include_tools_used_annotation(automatic_router, mock_local_agent):
    """History must never contain [Tools used: ...] — teaching the model to emit it."""
    mock_local_agent.run.return_value = {
        "speak": "Done.", "display": "Done.",
        "steps": [
            {"tool": "web_fetch", "input_summary": "...", "milestone": False},
            {"tool": "run_code", "input_summary": "...", "milestone": False},
        ],
    }
    with patch.object(automatic_router, "_classify", return_value={
        "can_handle_locally": True, "intent_class": "read_only", "reason": "test"
    }):
        automatic_router.process("fetch github issues")

    assistant_msg = automatic_router._history[-1]["content"]
    assert "[Tools used:" not in assistant_msg


def test_history_no_tools_suffix_when_no_steps(automatic_router, mock_local_agent):
    """Assistant history entry has no [Tools used] suffix when steps list is empty."""
    mock_local_agent.run.return_value = {
        "speak": "Done.", "display": "Done.", "steps": [],
    }
    with patch.object(automatic_router, "_classify", return_value={
        "can_handle_locally": True, "intent_class": "read_only", "reason": "test"
    }):
        automatic_router.process("hello")

    assistant_msg = automatic_router._history[-1]["content"]
    assert "[Tools used:" not in assistant_msg


# ── classifier shared http client ─────────────────────────────────────────────

def test_classify_reuses_shared_http_client(config):
    """Router._classify must use a shared httpx.Client, not create a new one each call."""
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    assert hasattr(r, "_http_client"), "Router must have a shared _http_client"
    assert r._http_client is not None


# ── _append_turn, compaction, classifier context ──────────────────────────────

def test_append_turn_stores_compressed_steps(automatic_router):
    result = {
        "speak": "Done.",
        "display": "Done with details.",
        "steps": [
            {"tool": "shell_run", "input_summary": "git log --oneline", "result_summary": "abc1234 fix: prevent crash"},
            {"tool": "file_read", "input_summary": "/tmp/x.py", "result_summary": "def main(): pass"},
        ],
    }
    automatic_router._append_turn("show me the log", result)
    assistant_content = automatic_router._history[-1]["content"]
    assert "shell_run" in assistant_content
    assert "abc1234 fix: prevent crash" in assistant_content
    assert "file_read" in assistant_content


def test_append_turn_skips_approval_required(automatic_router):
    result = {"approval_required": {"tool": "shell_run"}, "steps": []}
    automatic_router._append_turn("delete stuff", result)
    assert automatic_router._history == []


def test_append_turn_no_steps_stores_display_text(automatic_router):
    result = {"speak": "Hi there!", "display": "Hi there!", "steps": []}
    automatic_router._append_turn("hello", result)
    assert automatic_router._history[-1]["content"] == "Hi there!"


def test_classify_receives_history_context(config):
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    history = [
        {"role": "user", "content": "check CI"},
        {"role": "assistant", "content": "CI is failing on test_foo"},
    ]
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": '{"intent_class":"prepare","can_handle_locally":true}'}}]}
        return resp

    with patch.object(r._http_client, "post", side_effect=fake_post):
        r._classify("yes please", history=history)

    messages = captured["json"]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "yes please"


def test_compact_fires_when_tokens_exceed_threshold(automatic_router, mock_local_agent):
    """Compaction is deferred: _append_turn sets the flag; process() triggers it."""
    large_content = "x" * 20001  # ~5000 tokens
    automatic_router._history = [
        {"role": "user", "content": large_content},
        {"role": "assistant", "content": "response"},
    ]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Compact summary of the session.")]
    mock_local_agent.run.return_value = {"speak": "next", "display": "next", "steps": []}

    # First: _append_turn should set the deferred flag but NOT compact yet.
    automatic_router._append_turn("another command", {"speak": "ok", "display": "ok", "steps": []})
    assert automatic_router._needs_compaction is True
    assert len(automatic_router._history) == 4  # not compacted yet

    # Second: process() triggers the deferred compact at its start.
    with patch.object(automatic_router._anthropic_client.messages, "create", return_value=mock_response):
        automatic_router.process("next command")

    assert len(automatic_router._history) == 4  # 2 compact + 2 new from process()
    assert automatic_router._history[0]["content"] == "[Prior conversation compacted]"
    assert "Compact summary" in automatic_router._history[1]["content"]
    assert automatic_router._pending_compaction_notice is True


def test_compact_best_effort_on_failure(automatic_router):
    automatic_router._history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    original_history = list(automatic_router._history)
    with patch.object(automatic_router._anthropic_client.messages, "create", side_effect=Exception("network error")):
        automatic_router._compact()

    assert automatic_router._history == original_history
    assert automatic_router._pending_compaction_notice is False
    assert automatic_router._compact_failed is True  # circuit breaker set


def test_compaction_notice_emitted_on_next_process(automatic_router, mock_local_agent):
    automatic_router._pending_compaction_notice = True
    events = []
    mock_local_agent.run.return_value = {"speak": "ok", "display": "ok", "steps": []}
    with patch.object(automatic_router, "_classify", return_value={"intent_class": "read_only", "can_handle_locally": True}):
        automatic_router.process("do something", step_callback=events.append)

    assert any(e.get("type") == "compacted" for e in events)
    assert automatic_router._pending_compaction_notice is False


# ── PromptLoader + system prompt + user-message prefix ────────────────────────

@pytest.fixture
def mock_prompt_loader(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "base.md").write_text("base prompt {home}")
    (prompts / "local.md").write_text("local rules")
    refs = tmp_path / "refs"
    projects = tmp_path / "projects"
    return PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)


@pytest.fixture
def prefix_router(config, mock_prompt_loader):
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails, prompt_loader=mock_prompt_loader)
    r._local = MagicMock()
    r._local.run.return_value = {"speak": "done", "display": "done", "error": None}
    r._sonnet = MagicMock()
    r._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "prepare", "reason": "test"})
    return r


def test_user_message_includes_cwd_on_first_call(prefix_router):
    with patch("router.get_git_context", return_value=None):
        prefix_router.process("do something", cwd="/some/project")
    call_kwargs = prefix_router._local.run.call_args.kwargs
    assert "[cwd: /some/project]" in call_kwargs["user_text"]


def test_user_message_includes_date(prefix_router):
    with patch("router.get_git_context", return_value=None):
        prefix_router.process("do something", cwd="/some/project")
    call_kwargs = prefix_router._local.run.call_args.kwargs
    assert f"[Date: {date.today()}]" in call_kwargs["user_text"]


def test_git_context_injected_on_first_call(prefix_router):
    ctx = {"branch": "main", "commits": ["abc feat: x"], "remote": "https://github.com/u/r"}
    with patch("router.get_git_context", return_value=ctx):
        prefix_router.process("do something", cwd="/repo")
    call_kwargs = prefix_router._local.run.call_args.kwargs
    assert "branch=main" in call_kwargs["user_text"]


def test_git_context_not_repeated_when_unchanged(prefix_router):
    ctx = {"branch": "main", "commits": ["abc feat: x"], "remote": "https://github.com/u/r"}
    with patch("router.get_git_context", return_value=ctx):
        prefix_router.process("first", cwd="/repo")
        prefix_router._local.run.reset_mock()
        prefix_router.process("second", cwd="/repo")
    call_kwargs = prefix_router._local.run.call_args.kwargs
    assert "branch=main" not in call_kwargs["user_text"]


def test_git_context_re_injected_on_branch_change(prefix_router):
    ctx1 = {"branch": "main", "commits": ["abc feat: x"], "remote": None}
    ctx2 = {"branch": "feature/y", "commits": ["def feat: y"], "remote": None}
    with patch("router.get_git_context", side_effect=[ctx1, ctx2]):
        prefix_router.process("first", cwd="/repo")
        prefix_router._local.run.reset_mock()
        prefix_router.process("second", cwd="/repo")
    call_kwargs = prefix_router._local.run.call_args.kwargs
    assert "branch=feature/y" in call_kwargs["user_text"]


def test_system_prompt_passed_to_agent(prefix_router):
    with patch("router.get_git_context", return_value=None):
        prefix_router.process("do something", cwd="/repo")
    call_kwargs = prefix_router._local.run.call_args.kwargs
    assert call_kwargs.get("system_prompt") is not None
    assert "base prompt" in call_kwargs["system_prompt"]


def test_git_context_skipped_for_read_only_intent(prefix_router):
    prefix_router._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "read_only", "reason": "test"})
    ctx = {"branch": "main", "commits": ["abc feat: x"], "remote": None}
    with patch("router.get_git_context", return_value=ctx) as mock_gc:
        prefix_router.process("what time is it", cwd="/repo")
    mock_gc.assert_not_called()


# ── destructive intent approval gate ─────────────────────────────────────────

def test_destructive_intent_returns_approval_required(automatic_router, mock_local_agent):
    """A destructive command must pause and return approval_required before running."""
    automatic_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "deletes files"
    })
    result = automatic_router.process("delete all temp files", command_id="cmd-1")
    assert "approval_required" in result
    assert result["approval_required"]["category"] == "destructive_command"
    mock_local_agent.run.assert_not_called()


def test_destructive_intent_registers_resume_in_approval_store(automatic_router):
    """Paused destructive command must be registered so resume() can continue it."""
    import approval_store
    approval_store.clear()
    automatic_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "rm -rf"
    })
    automatic_router.process("wipe the build directory", command_id="cmd-2")
    assert approval_store.has("cmd-2")


def test_destructive_intent_resume_runs_agent(automatic_router, mock_local_agent):
    """After approval, resume() must run the agent and return an annotated result."""
    import approval_store
    approval_store.clear()
    mock_local_agent.run.return_value = {"speak": "Done.", "display": "Done.", "steps": []}
    automatic_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "deletes"
    })
    automatic_router.process("remove old logs", command_id="cmd-3")
    assert approval_store.has("cmd-3")

    result = automatic_router.resume("cmd-3")
    assert result["speak"] == "Done."
    assert result["_intent_class"] == "destructive"
    mock_local_agent.run.assert_called_once()
    assert not approval_store.has("cmd-3")


def test_non_destructive_intent_runs_without_gate(automatic_router, mock_local_agent):
    """Non-destructive commands must not be gated — agent runs immediately."""
    import approval_store
    approval_store.clear()
    automatic_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "prepare", "reason": "read-ish"
    })
    result = automatic_router.process("list all files", command_id="cmd-4")
    assert "approval_required" not in result
    mock_local_agent.run.assert_called_once()
    assert not approval_store.has("cmd-4")


def test_destructive_without_command_id_runs_immediately(automatic_router, mock_local_agent):
    """Without a command_id there's no way to resume, so run immediately (no gate)."""
    automatic_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "deletes"
    })
    result = automatic_router.process("delete build artifacts")  # no command_id
    assert "approval_required" not in result
    mock_local_agent.run.assert_called_once()


def test_destructive_gate_in_automatic_mode(config, mock_local_agent):
    """automatic routing must also gate destructive commands."""
    import approval_store
    approval_store.clear()
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._local = mock_local_agent
    r._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "rm"
    })
    result = r.process("clean the repo", command_id="cmd-5")
    assert "approval_required" in result
    mock_local_agent.run.assert_not_called()
    assert approval_store.has("cmd-5")


# ── router_passes_step_callback ───────────────────────────────────────────────

def test_router_passes_step_callback_to_agent(automatic_router):
    """Router forwards step_callback kwarg to agent.run()."""
    cb = MagicMock()
    with patch.object(automatic_router._local, "run", return_value={
        "speak": "ok", "display": "ok", "steps": []
    }) as mock_run:
        automatic_router.process("hello", step_callback=cb)
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs.get("step_callback") == cb


# ── intent_class destructive annotation ───────────────────────────────────────

def test_intent_class_destructive_annotated_in_metadata(config):
    """Destructive intent class should appear in response metadata."""
    config["local"]["routing_mode"] = "automatic"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_local = MagicMock()
    mock_local.run.return_value = {"speak": "Done.", "display": "Done."}
    r._local = mock_local

    classification = {"can_handle_locally": True, "intent_class": "destructive", "reason": "deletes file"}
    with patch.object(r, "_classify", return_value=classification):
        result = r.process("delete temp files", cwd=None)

    assert result.get("_intent_class") == "destructive"
