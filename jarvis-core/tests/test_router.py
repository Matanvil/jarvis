import pytest
from unittest.mock import MagicMock, patch
from guardrails import Guardrails
from ollama_agent import EscalateToCloud
from router import Router


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
    assert called_kwargs["memory_context"] == ""
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


def test_haiku_first_is_default_routing_mode(config):
    assert config["ollama"]["routing_mode"] == "haiku_first"


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
