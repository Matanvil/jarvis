import os
import pytest
from datetime import date
from unittest.mock import MagicMock, patch
from guardrails import Guardrails
from ollama_agent import EscalateToCloud
from router import Router
from prompt_loader import PromptLoader


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import config as cfg_module
    return cfg_module.load()


@pytest.fixture
def mock_ollama_agent():
    agent = MagicMock()
    agent.run.return_value = {"speak": "Done locally.", "display": "Done locally.", "error": None}
    return agent


@pytest.fixture
def mock_claude_agent():
    agent = MagicMock()
    agent.run.return_value = {"speak": "Done via Claude.", "display": "Done via Claude.", "error": None}
    return agent


@pytest.fixture
def router(config, mock_ollama_agent, mock_claude_agent):
    config["ollama"]["routing_mode"] = "ollama_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._ollama = mock_ollama_agent
    r._claude = mock_claude_agent
    r._sonnet = mock_claude_agent
    # Pre-flight classifier is mocked by default; individual tests can override
    r._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "read_only", "reason": "test"})
    return r


# ── routing_mode: ollama_first ────────────────────────────────────────────────

def test_ollama_first_uses_ollama_when_successful(router, mock_ollama_agent, mock_claude_agent):
    result = router.process("list Downloads", cwd=None)
    called_kwargs = mock_ollama_agent.run.call_args.kwargs
    assert called_kwargs["cwd"] is None
    assert "memory_context" not in called_kwargs  # now baked into user_text prefix
    assert isinstance(called_kwargs["history"], list)
    mock_claude_agent.run.assert_not_called()
    assert result["speak"] == "Done locally."


def test_ollama_first_falls_back_to_claude_when_ollama_unreachable(router, mock_ollama_agent, mock_claude_agent):
    """When Ollama raises EscalateToCloud in ollama_first mode, router falls through to Claude."""
    mock_ollama_agent.run.side_effect = EscalateToCloud("connection refused")
    result = router.process("open Safari", cwd=None)
    mock_claude_agent.run.assert_called_once()
    assert result["_agent"] == "claude"
    assert result["_escalated"] is True


# ── routing_mode: claude_only ─────────────────────────────────────────────────

def test_claude_only_skips_ollama(router, mock_ollama_agent, mock_claude_agent, config):
    config["ollama"]["routing_mode"] = "claude_only"
    result = router.process("any command", cwd=None)
    mock_ollama_agent.run.assert_not_called()
    mock_claude_agent.run.assert_called_once()
    assert result["speak"] == "Done via Claude."


# ── routing_mode: ollama_only ─────────────────────────────────────────────────

def test_ollama_only_never_calls_claude(router, mock_ollama_agent, mock_claude_agent, config):
    config["ollama"]["routing_mode"] = "ollama_only"
    mock_ollama_agent.run.return_value = {"speak": "Best I can do offline.", "display": "Best I can do offline."}
    result = router.process("search the web", cwd=None)
    mock_claude_agent.run.assert_not_called()
    assert result["speak"] == "Best I can do offline."


# ── analytics logging ─────────────────────────────────────────────────────────

def test_process_returns_agent_metadata(router):
    result = router.process("list Downloads", cwd=None)
    assert result.get("_agent") == "ollama"
    assert result.get("_escalated") is False
    assert isinstance(result.get("_response_ms"), int)


def test_process_returns_intent_class_in_metadata(router):
    """Intent class from pre-flight classifier should appear in response metadata."""
    result = router.process("list Downloads", cwd=None)
    assert result.get("_agent") == "ollama"
    assert result.get("_escalated") is False
    assert result.get("_intent_class") == "read_only"


def test_ollama_only_guard_records_escalation_truthfully(router, mock_ollama_agent, mock_claude_agent, config):
    """If OllamaAgent leaks EscalateToCloud in ollama_only mode, router must not call Claude
    and must record escalated=True with the suppression reason."""
    config["ollama"]["routing_mode"] = "ollama_only"
    mock_ollama_agent.run.side_effect = EscalateToCloud("unexpected reason")
    result = router.process("something", cwd=None)
    mock_claude_agent.run.assert_not_called()
    assert result.get("_escalated") is True
    assert "suppressed:ollama_only" in result.get("_escalation_reason", "")


def test_ollama_only_with_unreachable_ollama_does_not_call_claude(router, mock_ollama_agent, mock_claude_agent, config):
    """When Ollama is unreachable in ollama_only mode, Router must not escalate to Claude."""
    config["ollama"]["routing_mode"] = "ollama_only"
    mock_ollama_agent.run.side_effect = EscalateToCloud("Ollama unavailable: connection refused")
    result = router.process("open Safari", cwd=None)
    mock_claude_agent.run.assert_not_called()
    assert result.get("_escalated") is True
    assert "suppressed:ollama_only" in result.get("_escalation_reason", "")
    assert result.get("speak") is not None  # returned something, not None


# ── routing_mode validation ───────────────────────────────────────────────────

def test_invalid_routing_mode_logs_warning(config):
    """Invalid routing_mode should log a warning, not crash."""
    import logging
    config["ollama"]["routing_mode"] = "invalid_mode"
    guardrails = Guardrails(config)
    # Should not raise — warning goes to named logger (jarvis.errors), not root logger
    r = Router(config=config, guardrails=guardrails)
    assert r is not None


def test_valid_routing_mode_does_not_log_warning(config):
    """Valid routing_mode should not log a warning."""
    config["ollama"]["routing_mode"] = "claude_only"
    guardrails = Guardrails(config)
    with patch("logging.Logger.warning") as mock_warning:
        r = Router(config=config, guardrails=guardrails)
    mock_warning.assert_not_called()


# ── pre-flight classifier ──────────────────────────────────────────────────────

def test_classifier_routes_local_task_to_ollama(config):
    """read_only + can_handle_locally=True → OllamaAgent, no Claude call."""
    config["ollama"]["routing_mode"] = "ollama_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_ollama = MagicMock()
    mock_claude = MagicMock()
    mock_ollama.run.return_value = {"speak": "Done.", "display": "Done."}
    r._ollama = mock_ollama
    r._claude = mock_claude

    classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "file op"}
    with patch.object(r, "_classify", return_value=classification):
        result = r.process("list my files", cwd=None)

    mock_ollama.run.assert_called_once()
    mock_claude.run.assert_not_called()


def test_classifier_routes_complex_task_to_claude(config):
    """can_handle_locally=False → Agent (Claude), no Ollama call."""
    config["ollama"]["routing_mode"] = "ollama_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_ollama = MagicMock()
    mock_claude = MagicMock()
    mock_claude.run.return_value = {"speak": "Done via Claude.", "display": "Done."}
    r._ollama = mock_ollama
    r._claude = mock_claude

    classification = {"can_handle_locally": False, "intent_class": "complex_reasoning", "reason": "web search"}
    with patch.object(r, "_classify", return_value=classification):
        result = r.process("latest React news", cwd=None)

    mock_ollama.run.assert_not_called()
    mock_claude.run.assert_called_once()
    assert result.get("_escalated") is False   # direct routing, not escalation


def test_classifier_failure_falls_back_to_ollama_first(config):
    """If _classify raises, Router falls back to ollama_first behaviour."""
    config["ollama"]["routing_mode"] = "ollama_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_ollama = MagicMock()
    mock_claude = MagicMock()
    mock_ollama.run.return_value = {"speak": "Done.", "display": "Done."}
    r._ollama = mock_ollama
    r._claude = mock_claude

    with patch.object(r, "_classify", side_effect=Exception("classifier down")):
        result = r.process("open Safari", cwd=None)

    mock_ollama.run.assert_called_once()   # fell back gracefully


# ── routing_mode: haiku_first ─────────────────────────────────────────────────

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
def haiku_router(config, mock_ollama_agent, mock_haiku_agent, mock_sonnet_agent):
    config["ollama"]["routing_mode"] = "haiku_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._ollama = mock_ollama_agent
    r._haiku = mock_haiku_agent
    r._sonnet = mock_sonnet_agent
    r._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "read_only", "reason": "test"})
    return r


def test_haiku_first_uses_haiku_for_simple_tasks(haiku_router, mock_haiku_agent, mock_sonnet_agent, mock_ollama_agent):
    result = haiku_router.process("list my files")
    mock_haiku_agent.run.assert_called_once()
    mock_sonnet_agent.run.assert_not_called()
    mock_ollama_agent.run.assert_not_called()
    assert result["speak"] == "Done via Haiku."
    assert result["_model"] == "claude-haiku-4-5-20251001"


def test_haiku_first_uses_sonnet_for_complex_reasoning(haiku_router, mock_haiku_agent, mock_sonnet_agent):
    haiku_router._classify = MagicMock(return_value={
        "can_handle_locally": False, "intent_class": "complex_reasoning", "reason": "needs web"
    })
    result = haiku_router.process("what's the latest news on AI?")
    mock_sonnet_agent.run.assert_called_once()
    mock_haiku_agent.run.assert_not_called()
    assert result["_model"] == "claude-sonnet-4-6"


def test_config_fixture_default_routing_mode_is_local_first(config):
    assert config["ollama"]["routing_mode"] == "local_first"


def test_haiku_first_classifier_failure_falls_back_to_haiku(haiku_router, mock_haiku_agent, mock_sonnet_agent):
    haiku_router._classify = MagicMock(side_effect=Exception("classifier down"))
    result = haiku_router.process("hello")
    mock_haiku_agent.run.assert_called_once()
    mock_sonnet_agent.run.assert_not_called()


def test_router_passes_step_callback_to_agent(haiku_router):
    """Router forwards step_callback kwarg to agent.run()."""
    cb = MagicMock()
    with patch.object(haiku_router._haiku, "run", return_value={
        "speak": "ok", "display": "ok", "steps": []
    }) as mock_run:
        haiku_router.process("hello", step_callback=cb)
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs.get("step_callback") == cb


def test_intent_class_destructive_annotated_in_metadata(config):
    """Destructive intent class should appear in response metadata."""
    config["ollama"]["routing_mode"] = "ollama_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)

    mock_ollama = MagicMock()
    mock_claude = MagicMock()
    mock_ollama.run.return_value = {"speak": "Done.", "display": "Done."}
    r._ollama = mock_ollama
    r._claude = mock_claude

    classification = {"can_handle_locally": True, "intent_class": "destructive", "reason": "deletes file"}
    with patch.object(r, "_classify", return_value=classification):
        result = r.process("delete temp files", cwd=None)

    assert result.get("_intent_class") == "destructive"


# ── routing_mode: local_first ────────────────────────────────────────────────

@pytest.fixture
def local_first_router(config, mock_ollama_agent, mock_haiku_agent, mock_sonnet_agent):
    config["ollama"]["routing_mode"] = "local_first"
    config["ollama"]["model"] = "qwen-executor"
    config["ollama"]["executor_model"] = "supergemma4-test"
    config["ollama"]["classifier_model"] = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._ollama = mock_ollama_agent
    r._haiku = mock_haiku_agent
    r._sonnet = mock_sonnet_agent
    r._claude = mock_sonnet_agent
    return r


def test_local_first_uses_ollama_for_simple_tasks(local_first_router, mock_ollama_agent, mock_sonnet_agent):
    result = local_first_router.process("list my files")
    mock_ollama_agent.run.assert_called_once()
    mock_sonnet_agent.run.assert_not_called()
    assert result["_agent"] == "ollama"
    assert result["_model"] == "supergemma4-test"


def test_local_first_uses_sonnet_for_complex_reasoning(local_first_router, mock_ollama_agent, mock_sonnet_agent):
    local_first_router._classify = MagicMock(return_value={
        "can_handle_locally": False, "intent_class": "complex_reasoning", "reason": "needs web"
    })
    result = local_first_router.process("search the web for latest news")
    mock_ollama_agent.run.assert_not_called()
    mock_sonnet_agent.run.assert_called_once()
    assert result["_agent"] == "claude"
    assert result["_model"] == "claude-sonnet-4-6"


def test_local_first_stays_local_on_escalate(local_first_router, mock_ollama_agent, mock_sonnet_agent):
    """In local_first mode, OllamaAgent failure must return a graceful message — never call cloud."""
    from ollama_agent import EscalateToCloud
    local_first_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "read_only", "reason": "simple"
    })
    mock_ollama_agent.run.side_effect = EscalateToCloud("timeout")
    result = local_first_router.process("do something hard")
    mock_sonnet_agent.run.assert_not_called()
    assert result["_escalated"] is True
    assert "local_first" in result["_escalation_reason"]
    assert result["speak"] is not None


def test_local_first_classifier_failure_falls_back_to_ollama(local_first_router, mock_ollama_agent, mock_sonnet_agent):
    local_first_router._classify = MagicMock(side_effect=Exception("classifier down"))
    result = local_first_router.process("hello")
    mock_ollama_agent.run.assert_called_once()
    mock_sonnet_agent.run.assert_not_called()


def test_classifier_uses_classifier_model(config):
    """_classifier_model should use classifier_model key when present."""
    config["ollama"]["model"] = "qwen-executor"
    config["ollama"]["classifier_model"] = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    assert r._classifier_model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert r._ollama_model == "qwen-executor"


def test_default_routing_mode_is_local_first():
    """Default routing mode should be local_first."""
    from config import DEFAULTS
    assert DEFAULTS["ollama"]["routing_mode"] == "local_first"


def test_resume_returns_none_when_no_paused_run(haiku_router):
    import approval_store
    approval_store.clear()
    assert haiku_router.resume("nope") is None


def test_resume_invokes_stored_resumer_and_annotates(haiku_router):
    import approval_store
    approval_store.clear()
    approval_store.register(
        "c1",
        lambda step_callback=None: {"speak": "ok", "display": "ok", "steps": []},
        {"user_text": "hi", "agent": "ollama", "model": "m1"},
    )
    result = haiku_router.resume("c1")
    assert result["_agent"] == "ollama"
    assert result["_model"] == "m1"
    assert result["speak"] == "ok"
    assert not approval_store.has("c1")  # popped


def test_default_base_ollama_model_is_qwen36():
    """Default base Ollama model should align with the selected Qwen executor."""
    from config import DEFAULTS
    assert DEFAULTS["ollama"]["model"] == "qwen3.6:35b-a3b"


def test_classifier_model_in_defaults():
    """classifier_model should be present in DEFAULTS and point to jarvis-classifier."""
    from config import DEFAULTS
    assert DEFAULTS["ollama"]["classifier_model"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def test_local_first_annotates_result_with_executor_model_name(local_first_router, mock_ollama_agent):
    """_model in result should reflect the ollama.model (executor), not classifier_model."""
    result = local_first_router.process("list my files")
    assert result["_model"] == "supergemma4-test"  # matches what local_first_router fixture sets


# ── history tool summary ──────────────────────────────────────────────────────

def test_history_does_not_include_tools_used_annotation(haiku_router, mock_haiku_agent):
    """History must never contain [Tools used: ...] — teaching the model to emit it."""
    mock_haiku_agent.run.return_value = {
        "speak": "Done.", "display": "Done.",
        "steps": [
            {"tool": "web_fetch", "input_summary": "...", "milestone": False},
            {"tool": "run_code", "input_summary": "...", "milestone": False},
        ],
    }
    with patch.object(haiku_router, "_classify", return_value={
        "can_handle_locally": False, "intent_class": "read_only", "reason": "test"
    }):
        haiku_router.process("fetch github issues")

    assistant_msg = haiku_router._history[-1]["content"]
    assert "[Tools used:" not in assistant_msg


def test_history_no_tools_suffix_when_no_steps(haiku_router, mock_haiku_agent):
    """Assistant history entry has no [Tools used] suffix when steps list is empty."""
    mock_haiku_agent.run.return_value = {
        "speak": "Done.", "display": "Done.", "steps": [],
    }
    with patch.object(haiku_router, "_classify", return_value={
        "can_handle_locally": False, "intent_class": "read_only", "reason": "test"
    }):
        haiku_router.process("hello")

    assistant_msg = haiku_router._history[-1]["content"]
    assert "[Tools used:" not in assistant_msg


# ── classifier shared http client ─────────────────────────────────────────────

def test_classify_reuses_shared_http_client(config):
    """Router._classify must use a shared httpx.Client, not create a new one each call."""
    config["ollama"]["routing_mode"] = "local_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    assert hasattr(r, "_http_client"), "Router must have a shared _http_client"
    assert r._http_client is not None


# ── Task 9: _append_turn, compaction, classifier context ─────────────────────

def test_append_turn_stores_compressed_steps(haiku_router):
    result = {
        "speak": "Done.",
        "display": "Done with details.",
        "steps": [
            {"tool": "shell_run", "input_summary": "git log --oneline", "result_summary": "abc1234 fix: prevent crash"},
            {"tool": "file_read", "input_summary": "/tmp/x.py", "result_summary": "def main(): pass"},
        ],
    }
    haiku_router._append_turn("show me the log", result)
    assistant_content = haiku_router._history[-1]["content"]
    assert "shell_run" in assistant_content
    assert "abc1234 fix: prevent crash" in assistant_content
    assert "file_read" in assistant_content


def test_append_turn_skips_approval_required(haiku_router):
    result = {"approval_required": {"tool": "shell_run"}, "steps": []}
    haiku_router._append_turn("delete stuff", result)
    assert haiku_router._history == []


def test_append_turn_no_steps_stores_display_text(haiku_router):
    result = {"speak": "Hi there!", "display": "Hi there!", "steps": []}
    haiku_router._append_turn("hello", result)
    assert haiku_router._history[-1]["content"] == "Hi there!"


def test_classify_receives_history_context(config):
    config["ollama"]["routing_mode"] = "local_first"
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


def test_compact_fires_when_tokens_exceed_threshold(haiku_router, mock_haiku_agent):
    """Compaction is deferred: _append_turn sets the flag; process() triggers it."""
    large_content = "x" * 20001  # ~5000 tokens
    haiku_router._history = [
        {"role": "user", "content": large_content},
        {"role": "assistant", "content": "response"},
    ]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Compact summary of the session.")]
    mock_haiku_agent.run.return_value = {"speak": "next", "display": "next", "steps": []}

    # First: _append_turn should set the deferred flag but NOT compact yet.
    haiku_router._append_turn("another command", {"speak": "ok", "display": "ok", "steps": []})
    assert haiku_router._needs_compaction is True
    assert len(haiku_router._history) == 4  # not compacted yet

    # Second: process() triggers the deferred compact at its start.
    with patch.object(haiku_router._anthropic_client.messages, "create", return_value=mock_response):
        haiku_router.process("next command")

    assert len(haiku_router._history) == 4  # 2 compact + 2 new from process()
    assert haiku_router._history[0]["content"] == "[Prior conversation compacted]"
    assert "Compact summary" in haiku_router._history[1]["content"]
    assert haiku_router._pending_compaction_notice is True


def test_compact_best_effort_on_failure(haiku_router):
    haiku_router._history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    original_history = list(haiku_router._history)
    with patch.object(haiku_router._anthropic_client.messages, "create", side_effect=Exception("network error")):
        haiku_router._compact()

    assert haiku_router._history == original_history
    assert haiku_router._pending_compaction_notice is False
    assert haiku_router._compact_failed is True  # circuit breaker set


def test_compaction_notice_emitted_on_next_process(haiku_router, mock_haiku_agent):
    haiku_router._pending_compaction_notice = True
    events = []
    mock_haiku_agent.run.return_value = {"speak": "ok", "display": "ok", "steps": []}
    with patch.object(haiku_router, "_classify", return_value={"intent_class": "read_only", "can_handle_locally": True}):
        haiku_router.process("do something", step_callback=events.append)

    assert any(e.get("type") == "compacted" for e in events)
    assert haiku_router._pending_compaction_notice is False


# ── Task 6: PromptLoader + system prompt + user-message prefix ────────────────

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
    config["ollama"]["routing_mode"] = "local_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails, prompt_loader=mock_prompt_loader)
    r._ollama = MagicMock()
    r._ollama.run.return_value = {"speak": "done", "display": "done", "error": None}
    r._sonnet = MagicMock()
    r._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "prepare", "reason": "test"})
    return r


def test_user_message_includes_cwd_on_first_call(prefix_router):
    with patch("router.get_git_context", return_value=None):
        prefix_router.process("do something", cwd="/some/project")
    call_kwargs = prefix_router._ollama.run.call_args.kwargs
    assert "[cwd: /some/project]" in call_kwargs["user_text"]


def test_user_message_includes_date(prefix_router):
    with patch("router.get_git_context", return_value=None):
        prefix_router.process("do something", cwd="/some/project")
    call_kwargs = prefix_router._ollama.run.call_args.kwargs
    assert f"[Date: {date.today()}]" in call_kwargs["user_text"]


def test_git_context_injected_on_first_call(prefix_router):
    ctx = {"branch": "main", "commits": ["abc feat: x"], "remote": "https://github.com/u/r"}
    with patch("router.get_git_context", return_value=ctx):
        prefix_router.process("do something", cwd="/repo")
    call_kwargs = prefix_router._ollama.run.call_args.kwargs
    assert "branch=main" in call_kwargs["user_text"]


def test_git_context_not_repeated_when_unchanged(prefix_router):
    ctx = {"branch": "main", "commits": ["abc feat: x"], "remote": "https://github.com/u/r"}
    with patch("router.get_git_context", return_value=ctx):
        prefix_router.process("first", cwd="/repo")
        prefix_router._ollama.run.reset_mock()
        prefix_router.process("second", cwd="/repo")
    call_kwargs = prefix_router._ollama.run.call_args.kwargs
    assert "branch=main" not in call_kwargs["user_text"]


def test_git_context_re_injected_on_branch_change(prefix_router):
    ctx1 = {"branch": "main", "commits": ["abc feat: x"], "remote": None}
    ctx2 = {"branch": "feature/y", "commits": ["def feat: y"], "remote": None}
    with patch("router.get_git_context", side_effect=[ctx1, ctx2]):
        prefix_router.process("first", cwd="/repo")
        prefix_router._ollama.run.reset_mock()
        prefix_router.process("second", cwd="/repo")
    call_kwargs = prefix_router._ollama.run.call_args.kwargs
    assert "branch=feature/y" in call_kwargs["user_text"]


def test_system_prompt_passed_to_agent(prefix_router):
    with patch("router.get_git_context", return_value=None):
        prefix_router.process("do something", cwd="/repo")
    call_kwargs = prefix_router._ollama.run.call_args.kwargs
    assert call_kwargs.get("system_prompt") is not None
    assert "base prompt" in call_kwargs["system_prompt"]


def test_git_context_skipped_for_read_only_intent(prefix_router):
    prefix_router._classify = MagicMock(return_value={"can_handle_locally": True, "intent_class": "read_only", "reason": "test"})
    ctx = {"branch": "main", "commits": ["abc feat: x"], "remote": None}
    with patch("router.get_git_context", return_value=ctx) as mock_gc:
        prefix_router.process("what time is it", cwd="/repo")
    mock_gc.assert_not_called()


# ── destructive intent approval gate ─────────────────────────────────────────

def test_destructive_intent_returns_approval_required(haiku_router, mock_haiku_agent):
    """A destructive command must pause and return approval_required before running."""
    haiku_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "deletes files"
    })
    result = haiku_router.process("delete all temp files", command_id="cmd-1")
    assert "approval_required" in result
    assert result["approval_required"]["category"] == "destructive_command"
    mock_haiku_agent.run.assert_not_called()


def test_destructive_intent_registers_resume_in_approval_store(haiku_router):
    """Paused destructive command must be registered so resume() can continue it."""
    import approval_store
    approval_store.clear()
    haiku_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "rm -rf"
    })
    haiku_router.process("wipe the build directory", command_id="cmd-2")
    assert approval_store.has("cmd-2")


def test_destructive_intent_resume_runs_agent(haiku_router, mock_haiku_agent):
    """After approval, resume() must run the agent and return an annotated result."""
    import approval_store
    approval_store.clear()
    mock_haiku_agent.run.return_value = {"speak": "Done.", "display": "Done.", "steps": []}
    haiku_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "deletes"
    })
    haiku_router.process("remove old logs", command_id="cmd-3")
    assert approval_store.has("cmd-3")

    result = haiku_router.resume("cmd-3")
    assert result["speak"] == "Done."
    assert result["_intent_class"] == "destructive"
    mock_haiku_agent.run.assert_called_once()
    assert not approval_store.has("cmd-3")


def test_non_destructive_intent_runs_without_gate(haiku_router, mock_haiku_agent):
    """Non-destructive commands must not be gated — agent runs immediately."""
    import approval_store
    approval_store.clear()
    haiku_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "prepare", "reason": "read-ish"
    })
    result = haiku_router.process("list all files", command_id="cmd-4")
    assert "approval_required" not in result
    mock_haiku_agent.run.assert_called_once()
    assert not approval_store.has("cmd-4")


def test_destructive_without_command_id_runs_immediately(haiku_router, mock_haiku_agent):
    """Without a command_id there's no way to resume, so run immediately (no gate)."""
    haiku_router._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "deletes"
    })
    result = haiku_router.process("delete build artifacts")  # no command_id
    assert "approval_required" not in result
    mock_haiku_agent.run.assert_called_once()


def test_destructive_gate_in_ollama_first_mode(config, mock_ollama_agent):
    """ollama_first routing must also gate destructive commands."""
    import approval_store
    approval_store.clear()
    config["ollama"]["routing_mode"] = "ollama_first"
    guardrails = Guardrails(config)
    r = Router(config=config, guardrails=guardrails)
    r._ollama = mock_ollama_agent
    r._classify = MagicMock(return_value={
        "can_handle_locally": True, "intent_class": "destructive", "reason": "rm"
    })
    result = r.process("clean the repo", command_id="cmd-5")
    assert "approval_required" in result
    mock_ollama_agent.run.assert_not_called()
    assert approval_store.has("cmd-5")
