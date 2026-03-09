import pytest
import subprocess
from unittest.mock import patch, MagicMock
from tools.macos import MacOSTool


def test_open_app_builds_correct_command():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = tool.open_app("Safari")
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "Safari" in " ".join(call_args)


def test_open_app_returns_error_on_failure():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Application not found")
        result = tool.open_app("FakeApp")
    assert result["success"] is False
    assert result["error"] is not None


def test_run_applescript_returns_output():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="result text\n", stderr="")
        result = tool.run_applescript('return "hello"')
    assert result["output"] == "result text\n"
    assert result["error"] is None


def test_run_applescript_returns_error_on_failure():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="script error")
        result = tool.run_applescript("invalid script")
    assert result["output"] is None
    assert "script error" in result["error"]


def test_run_applescript_timeout():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=15)):
        result = tool.run_applescript("delay 100")
    assert result["output"] is None
    assert "timed out" in result["error"].lower()


def test_notify_sends_notification():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = tool.notify("Test Title", "Test Body")
    assert result["success"] is True
    call_args = mock_run.call_args[0][0]
    assert any("display notification" in str(a) for a in call_args)


def test_notify_sanitizes_injection():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tool.notify('evil" & (do shell script "rm -rf /") & "', "body")
    call_args = mock_run.call_args[0][0]
    script = " ".join(call_args)
    # The raw injection string must not appear verbatim in the script
    assert 'do shell script' not in script or '\\"' in script


def test_set_volume_calls_applescript():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tool.set_volume(50)
    call_args = mock_run.call_args[0][0]
    assert any("50" in str(a) for a in call_args)


def test_set_volume_clamps_out_of_range():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tool.set_volume(150)
    call_args = mock_run.call_args[0][0]
    assert any("100" in str(a) for a in call_args)
    assert not any("150" in str(a) for a in call_args)


def test_get_frontmost_app_calls_applescript():
    tool = MacOSTool()
    with patch("tools.macos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Xcode\n", stderr="")
        result = tool.get_frontmost_app()
    assert result["output"] == "Xcode\n"
    assert result["error"] is None
