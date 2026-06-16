# jarvis-core/tests/test_git_context.py
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from git_context import get_git_context


def _make_run(stdout_map: dict):
    """Return a mock subprocess.run that returns stdout from stdout_map keyed by args[0][-1]."""
    def _run(args, **kwargs):
        key = args[-1]  # last arg is the git subcommand / flag
        m = MagicMock()
        m.returncode = 0
        m.stdout = stdout_map.get(key, "")
        return m
    return _run


def test_returns_branch_commits_remote(tmp_path):
    responses = {
        "--show-current": "main\n",
        "-3": "abc1234 feat: add thing\ndef5678 fix: bug\n",
        "origin": "https://github.com/Matanvil/jarvis\n",
    }
    with patch("git_context.subprocess.run", side_effect=_make_run(responses)):
        result = get_git_context(str(tmp_path))
    assert result is not None
    assert result["branch"] == "main"
    assert len(result["commits"]) == 2
    assert result["commits"][0] == "abc1234 feat: add thing"
    assert result["remote"] == "https://github.com/Matanvil/jarvis"


def test_returns_none_when_not_a_git_repo(tmp_path):
    with patch("git_context.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(128, "git")
        result = get_git_context(str(tmp_path))
    assert result is None


def test_returns_none_on_any_subprocess_failure(tmp_path):
    with patch("git_context.subprocess.run", side_effect=Exception("boom")):
        result = get_git_context(str(tmp_path))
    assert result is None


def test_missing_remote_still_returns_result(tmp_path):
    def _run(args, **kwargs):
        m = MagicMock()
        if "get-url" in args:
            m.returncode = 128
            m.stdout = ""
        else:
            m.returncode = 0
            m.stdout = "main\n" if "--show-current" in args else "abc1234 commit\n"
        return m
    with patch("git_context.subprocess.run", side_effect=_run):
        result = get_git_context(str(tmp_path))
    assert result is not None
    assert result["branch"] == "main"
    assert result.get("remote") is None
