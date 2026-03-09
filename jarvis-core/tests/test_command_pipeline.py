import time
import pytest
from unittest.mock import MagicMock
from command_pipeline import CommandPipeline, JarvisCommand, CommandStatus


@pytest.fixture
def mock_router():
    r = MagicMock()
    r.process.return_value = {"speak": "Done.", "display": "Done.", "_agent": "ollama"}
    return r


@pytest.fixture
def pipeline(mock_router):
    return CommandPipeline(router=mock_router)


# ── JarvisCommand ─────────────────────────────────────────────────────────────

def test_command_has_unique_id(pipeline):
    cmd1 = pipeline._create_command("hello", cwd=None, source="hotkey")
    cmd2 = pipeline._create_command("world", cwd=None, source="hotkey")
    assert cmd1.id != cmd2.id


def test_command_initial_status_is_received(pipeline):
    cmd = pipeline._create_command("test", cwd=None, source="voice")
    assert cmd.status == CommandStatus.RECEIVED


# ── submit: happy path ────────────────────────────────────────────────────────

def test_submit_returns_result_with_command_id(pipeline, mock_router):
    result = pipeline.submit("open Safari", cwd=None, source="hotkey")
    assert "command_id" in result
    assert result["speak"] == "Done."


def test_submit_calls_router_with_text_and_cwd(pipeline, mock_router):
    pipeline.submit("list files", cwd="/my/project", source="hotkey")
    mock_router.process.assert_called_once_with("list files", cwd="/my/project", memory_context="")


def test_submit_marks_command_completed(pipeline):
    result = pipeline.submit("do something", cwd=None, source="hotkey")
    cmd_id = result["command_id"]
    cmd = pipeline.get(cmd_id)
    assert cmd.status == CommandStatus.COMPLETED


# ── busy lock ─────────────────────────────────────────────────────────────────

def test_submit_while_busy_returns_busy_response(pipeline, mock_router):
    pipeline._executing = True
    pipeline._current_command_id = "fake-id"
    result = pipeline.submit("another command", cwd=None, source="hotkey")
    assert result.get("busy") is True
    assert result.get("command_id") == "fake-id"
    mock_router.process.assert_not_called()


def test_submit_releases_lock_after_completion(pipeline):
    pipeline.submit("do something", cwd=None, source="hotkey")
    assert pipeline._executing is False


def test_submit_releases_lock_on_router_exception(pipeline, mock_router):
    mock_router.process.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        pipeline.submit("bad command", cwd=None, source="hotkey")
    assert pipeline._executing is False


# ── cancel ────────────────────────────────────────────────────────────────────

def test_cancel_executing_command_releases_lock(pipeline):
    pipeline._executing = True
    pipeline._current_command_id = "abc"
    cmd = pipeline._create_command("running task", cwd=None, source="hotkey")
    cmd.id = "abc"
    cmd.status = CommandStatus.EXECUTING
    pipeline._registry["abc"] = cmd

    result = pipeline.cancel("abc")
    assert result["cancelled"] is True
    assert pipeline._executing is False
    assert pipeline._registry["abc"].status == CommandStatus.CANCELLED


def test_cancel_unknown_id_returns_not_found(pipeline):
    result = pipeline.cancel("nonexistent")
    assert result.get("error") == "not_found"


# ── abort ─────────────────────────────────────────────────────────────────────

def test_abort_force_releases_lock(pipeline):
    pipeline._executing = True
    pipeline._current_command_id = "stuck"
    result = pipeline.abort()
    assert result["lock_released"] is True
    assert pipeline._executing is False


# ── registry ──────────────────────────────────────────────────────────────────

def test_list_returns_recent_commands(pipeline):
    pipeline.submit("cmd 1", cwd=None, source="hotkey")
    pipeline.submit("cmd 2", cwd=None, source="hotkey")
    listing = pipeline.list_recent()
    assert len(listing) == 2


def test_get_returns_none_for_unknown_id(pipeline):
    assert pipeline.get("nonexistent") is None
