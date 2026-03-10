import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from command_pipeline import CommandPipeline
import telegram_state


@pytest.fixture(autouse=True)
def clean_telegram_state():
    telegram_state.reset_state()
    yield
    telegram_state.reset_state()


@pytest.fixture
def client_and_agent():
    mock_router = MagicMock()
    mock_router.process.return_value = {"speak": "Done!", "display": "Done!"}
    mock_guardrails = MagicMock()
    mock_loggers = MagicMock()
    mock_pipeline = CommandPipeline(router=mock_router)

    with patch("server._pipeline", mock_pipeline), \
         patch("server._guardrails", mock_guardrails), \
         patch("server._loggers", mock_loggers):
        import server
        client = TestClient(server.app, raise_server_exceptions=True)
        yield client, mock_router, mock_guardrails


def test_health_endpoint(client_and_agent):
    client, _, _ = client_and_agent
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_command_endpoint_returns_speak_and_display(client_and_agent):
    client, _, _ = client_and_agent
    response = client.post("/command", json={"text": "create a test file"})
    assert response.status_code == 200
    data = response.json()
    assert "speak" in data
    assert "display" in data


def test_command_endpoint_calls_agent_with_cwd(client_and_agent):
    client, mock_router, _ = client_and_agent
    client.post("/command", json={"text": "open safari", "cwd": "/my/project"})
    mock_router.process.assert_called_once_with("open safari", cwd="/my/project", memory_context="", source="hotkey")


def test_command_endpoint_cwd_defaults_to_none(client_and_agent):
    client, mock_router, _ = client_and_agent
    client.post("/command", json={"text": "hello"})
    mock_router.process.assert_called_once_with("hello", cwd=None, memory_context="", source="hotkey")


def test_approve_endpoint_trusts_session(client_and_agent):
    client, _, mock_guardrails = client_and_agent
    response = client.post("/approve", json={
        "tool_use_id": "abc",
        "approved": True,
        "trust_session": True,
        "category": "delete_files",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["acknowledged"] is True
    assert data["next_action"] == "reissue_command"
    mock_guardrails.trust_for_session.assert_called_once_with("delete_files")


def test_approve_without_trust_session_does_not_call_trust(client_and_agent):
    client, _, mock_guardrails = client_and_agent
    client.post("/approve", json={"tool_use_id": "abc", "approved": True})
    mock_guardrails.trust_for_session.assert_not_called()


def test_approve_denied_returns_cancelled(client_and_agent):
    client, _, _ = client_and_agent
    response = client.post("/approve", json={"tool_use_id": "abc", "approved": False})
    assert response.json()["next_action"] == "cancelled"


def test_command_returns_friendly_error_on_agent_exception(client_and_agent):
    client, mock_router, _ = client_and_agent
    mock_router.process.side_effect = RuntimeError("something exploded")
    response = client.post("/command", json={"text": "do something"})
    assert response.status_code == 200
    data = response.json()
    assert data["speak"] is not None
    assert data["speak"] == data["display"]
    assert data["steps"] == []


def test_command_response_includes_agent_metadata(client_and_agent):
    """Router metadata fields should pass through the /command response."""
    client, _, _ = client_and_agent
    with patch("server._pipeline") as mock_pipeline:
        mock_pipeline.submit.return_value = {
            "speak": "Done.", "display": "Done.",
            "_agent": "ollama", "_model": "mistral:latest",
            "_escalated": False, "_escalation_reason": None, "_response_ms": 42,
            "command_id": "test-id",
        }
        resp = client.post("/command", json={"text": "open Safari"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["speak"] == "Done."
    assert data["_agent"] == "ollama"


def test_command_logs_agent_analytics(client_and_agent):
    """Analytics log should include agent and escalated fields."""
    client, _, _ = client_and_agent
    import server as srv
    import logger as logger_module
    with patch("server._pipeline") as mock_pipeline, \
         patch.object(logger_module, "log_analytics") as mock_log_analytics:
        mock_pipeline.submit.return_value = {
            "speak": "Done.", "display": "Done.",
            "_agent": "claude", "_model": "claude-sonnet-4-6",
            "_escalated": True, "_escalation_reason": "web search", "_response_ms": 800,
            "command_id": "test-id",
        }
        resp = client.post("/command", json={"text": "search the web"})
    assert resp.status_code == 200
    mock_log_analytics.assert_called_once()
    analytics_payload = mock_log_analytics.call_args.args[2]
    assert analytics_payload["agent"] == "claude"
    assert analytics_payload["escalated"] is True
    assert analytics_payload["escalation_reason"] == "web search"
    assert analytics_payload["agent_response_ms"] == 800


def test_command_response_includes_command_id(client_and_agent):
    client, mock_router, _ = client_and_agent
    import server as srv
    from unittest.mock import patch as _patch
    with _patch("server._pipeline") as mock_pipeline:
        mock_pipeline.submit.return_value = {
            "speak": "Done.", "display": "Done.",
            "command_id": "test-uuid-123",
            "_agent": "ollama",
        }
        resp = client.post("/command", json={"text": "open Safari"})
    assert resp.status_code == 200
    assert resp.json()["command_id"] == "test-uuid-123"


def test_commands_list_endpoint(client_and_agent):
    client, _, _ = client_and_agent
    from unittest.mock import patch as _patch
    with _patch("server._pipeline") as mock_pipeline:
        mock_pipeline.list_recent.return_value = [
            {"id": "abc", "source": "hotkey", "raw_input": "test", "status": "COMPLETED",
             "created_at": 0.0, "completed_at": 1.0}
        ]
        resp = client.get("/commands")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_command_abort_endpoint(client_and_agent):
    client, _, _ = client_and_agent
    from unittest.mock import patch as _patch
    with _patch("server._pipeline") as mock_pipeline:
        mock_pipeline.abort.return_value = {"lock_released": True}
        resp = client.post("/commands/abort")
    assert resp.status_code == 200
    assert resp.json()["lock_released"] is True


def test_command_cancel_endpoint(client_and_agent):
    client, _, _ = client_and_agent
    from unittest.mock import patch as _patch
    with _patch("server._pipeline") as mock_pipeline:
        mock_pipeline.cancel.return_value = {"cancelled": True, "command_id": "xyz"}
        resp = client.post("/commands/xyz/cancel")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is True


def test_busy_response_when_pipeline_locked(client_and_agent):
    client, _, _ = client_and_agent
    from unittest.mock import patch as _patch
    with _patch("server._pipeline") as mock_pipeline:
        mock_pipeline.submit.return_value = {"busy": True, "command_id": "running-id"}
        resp = client.post("/command", json={"text": "new command"})
    assert resp.status_code == 200
    assert resp.json()["busy"] is True


def test_get_command_by_id_returns_command(client_and_agent):
    client, _, _ = client_and_agent
    from unittest.mock import patch as _patch
    from command_pipeline import JarvisCommand, CommandStatus
    import time
    fake_cmd = JarvisCommand(
        id="abc-123", source="hotkey", raw_input="open Safari",
        cwd=None, status=CommandStatus.COMPLETED,
        created_at=0.0, completed_at=1.0,
    )
    with _patch("server._pipeline") as mock_pipeline:
        mock_pipeline.get.return_value = fake_cmd
        resp = client.get("/commands/abc-123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "abc-123"
    assert data["raw_input"] == "open Safari"
    assert data["status"] == "COMPLETED"


def test_get_command_by_id_returns_404_for_unknown(client_and_agent):
    client, _, _ = client_and_agent
    from unittest.mock import patch as _patch
    with _patch("server._pipeline") as mock_pipeline:
        mock_pipeline.get.return_value = None
        resp = client.get("/commands/nonexistent")
    assert resp.status_code == 404


def test_empty_cwd_string_normalized_to_none(client_and_agent):
    """If Swift sends cwd='', it should be treated as None, not passed as empty string."""
    client, mock_router, _ = client_and_agent
    client.post("/command", json={"text": "hello", "cwd": ""})
    mock_router.process.assert_called_once_with("hello", cwd=None, memory_context="", source="hotkey")


def test_get_config_redacts_api_keys(client_and_agent, tmp_path):
    """GET /config must not expose values of keys ending in _key or _secret."""
    import server as srv
    import config as cfg_module
    with patch.object(cfg_module, "load", return_value={
        "anthropic_api_key": "sk-real-secret",
        "brave_api_key": "brave-real-secret",
        "server_port": 8765,
    }):
        resp = client_and_agent[0].get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["anthropic_api_key"] == "***"
    assert data["brave_api_key"] == "***"
    assert data["server_port"] == 8765


def test_get_config_redacts_nested_bot_token(client_and_agent):
    """GET /config must redact telegram.bot_token (nested sensitive key ending in _token)."""
    import config as cfg_module
    with patch.object(cfg_module, "load", return_value={
        "server_port": 8765,
        "telegram": {
            "bot_token": "real-bot-token-123",
            "allowed_user_id": 42,
        },
    }):
        resp = client_and_agent[0].get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["telegram"]["bot_token"] == "***"
    assert data["telegram"]["allowed_user_id"] == 42
    assert data["server_port"] == 8765


def test_config_post_deep_merges_guardrails(client_and_agent, tmp_path):
    import server as srv
    import config as cfg_module

    config_path = tmp_path / "config.json"
    original = {
        "anthropic_api_key": "sk-test",
        "guardrails": {"run_shell": "auto_allow", "delete_files": "require_approval"},
    }
    import json
    config_path.write_text(json.dumps(original))

    with patch.object(cfg_module, "load", return_value=dict(original)), \
         patch.object(cfg_module, "save") as mock_save:
        client_and_agent[0].post("/config", json={"guardrails": {"delete_files": "auto_allow"}})
        saved = mock_save.call_args[0][0]
        # run_shell must survive the partial update
        assert saved["guardrails"]["run_shell"] == "auto_allow"
        assert saved["guardrails"]["delete_files"] == "auto_allow"


def test_approve_classify_yes(client_and_agent):
    """'yes' should classify as approved=True."""
    with patch("server._classify_approval", return_value=True):
        resp = client_and_agent[0].post("/approve/classify", json={"text": "yes go ahead"})
    assert resp.status_code == 200
    assert resp.json()["approved"] is True


def test_approve_classify_no(client_and_agent):
    """'no' should classify as approved=False."""
    with patch("server._classify_approval", return_value=False):
        resp = client_and_agent[0].post("/approve/classify", json={"text": "no cancel that"})
    assert resp.json()["approved"] is False


def test_approve_classify_unclear(client_and_agent):
    """Unclear text should return approved=None."""
    with patch("server._classify_approval", return_value=None):
        resp = client_and_agent[0].post("/approve/classify", json={"text": "um what"})
    assert resp.json()["approved"] is None


# --- Convenience fixture that exposes just the TestClient ---
@pytest.fixture
def client(client_and_agent):
    return client_and_agent[0]


# --- Telegram /away endpoint and auto-disable tests ---

def test_telegram_away_endpoint_sets_true(client):
    resp = client.post("/telegram/away", json={"away": True})
    assert resp.status_code == 200
    assert telegram_state.get_state().away is True


def test_telegram_away_endpoint_sets_false(client):
    telegram_state.get_state().away = True
    resp = client.post("/telegram/away", json={"away": False})
    assert resp.status_code == 200
    assert telegram_state.get_state().away is False


def test_hotkey_command_while_away_auto_disables(client):
    telegram_state.get_state().away = True
    telegram_state.get_state().chat_id = 12345
    with patch("server.notify", new_callable=AsyncMock) as mock_notify:
        resp = client.post("/command", json={"text": "list files", "source": "hotkey"})
    assert resp.status_code == 200
    assert telegram_state.get_state().away is False
    mock_notify.assert_awaited_once_with("🟢 Jarvis back at the Mac — away mode off")


def test_telegram_command_does_not_disable_away(client):
    telegram_state.get_state().away = True
    with patch("server.notify", new_callable=AsyncMock):
        resp = client.post("/command", json={"text": "list files", "source": "telegram"})
    assert resp.status_code == 200
    assert telegram_state.get_state().away is True


# --- Scheduler endpoint tests ---

from dataclasses import asdict
from datetime import datetime, timezone
import scheduler as sched_module
from schedule_store import Schedule


@pytest.fixture
def sample_schedule():
    return Schedule(
        id="abc123",
        label="morning summary",
        command="summarise my calendar",
        schedule_type="recurring",
        cron="0 9 * * *",
        run_at_iso=None,
        enabled=True,
        created_at=datetime.now(timezone.utc).isoformat(),
        output="telegram",
    )


@pytest.fixture
def mock_scheduler(sample_schedule):
    s = MagicMock()
    s.list.return_value = [sample_schedule]
    s.create.return_value = sample_schedule
    s.delete.return_value = True
    s.pause.return_value = sample_schedule
    s.resume.return_value = sample_schedule
    return s


def test_get_schedules(client, mock_scheduler):
    with patch.object(sched_module, "_scheduler", mock_scheduler):
        resp = client.get("/schedules")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["schedules"]) == 1
    assert data["schedules"][0]["id"] == "abc123"


def test_create_schedule(client, mock_scheduler):
    with patch.object(sched_module, "_scheduler", mock_scheduler):
        resp = client.post("/schedules", json={
            "command": "summarise my calendar",
            "label": "morning summary",
            "schedule_type": "recurring",
            "cron": "0 9 * * *",
            "run_at_iso": None,
        })
    assert resp.status_code == 200
    assert resp.json()["id"] == "abc123"


def test_delete_schedule(client, mock_scheduler):
    with patch.object(sched_module, "_scheduler", mock_scheduler):
        resp = client.delete("/schedules/abc123")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_delete_schedule_not_found(client, mock_scheduler):
    mock_scheduler.delete.return_value = False
    with patch.object(sched_module, "_scheduler", mock_scheduler):
        resp = client.delete("/schedules/bad")
    assert resp.status_code == 404


def test_patch_schedule_pause(client, mock_scheduler):
    with patch.object(sched_module, "_scheduler", mock_scheduler):
        resp = client.patch("/schedules/abc123", json={"enabled": False})
    assert resp.status_code == 200
    mock_scheduler.pause.assert_called_once_with("abc123")


def test_patch_schedule_resume(client, mock_scheduler):
    with patch.object(sched_module, "_scheduler", mock_scheduler):
        resp = client.patch("/schedules/abc123", json={"enabled": True})
    assert resp.status_code == 200
    mock_scheduler.resume.assert_called_once_with("abc123")


def test_get_schedules_no_scheduler(client):
    with patch.object(sched_module, "_scheduler", None):
        resp = client.get("/schedules")
    assert resp.status_code == 200
    assert resp.json() == {"schedules": []}


def test_create_schedule_no_scheduler(client):
    with patch.object(sched_module, "_scheduler", None):
        resp = client.post("/schedules", json={
            "command": "cmd",
            "label": "label",
            "schedule_type": "recurring",
            "cron": "0 9 * * *",
            "run_at_iso": None,
        })
    assert resp.status_code == 503


def test_patch_schedule_not_found(client, mock_scheduler):
    mock_scheduler.pause.return_value = None
    with patch.object(sched_module, "_scheduler", mock_scheduler):
        resp = client.patch("/schedules/bad", json={"enabled": False})
    assert resp.status_code == 404
