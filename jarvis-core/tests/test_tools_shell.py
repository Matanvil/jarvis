import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from tools.shell import ShellTool


def test_run_simple_command():
    tool = ShellTool()
    result = tool.run("echo hello")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert result["stderr"] == ""


def test_run_failing_command():
    tool = ShellTool()
    result = tool.run("ls /nonexistent_path_xyz")
    assert result["exit_code"] != 0
    assert result["stderr"] != ""


def test_create_directory(tmp_path):
    tool = ShellTool()
    new_dir = tmp_path / "testdir"
    result = tool.run(f"mkdir {new_dir}")
    assert result["exit_code"] == 0
    assert new_dir.exists()


def test_write_file(tmp_path):
    tool = ShellTool()
    target = tmp_path / "hello.txt"
    result = tool.write_file(str(target), "hello world")
    assert result["success"] is True
    assert target.read_text() == "hello world"


def test_read_file(tmp_path):
    tool = ShellTool()
    target = tmp_path / "data.txt"
    target.write_text("some content")
    result = tool.read_file(str(target))
    assert result["content"] == "some content"


def test_read_missing_file():
    tool = ShellTool()
    result = tool.read_file("/nonexistent/file.txt")
    assert result["error"] is not None


def test_run_with_cwd(tmp_path):
    tool = ShellTool()
    (tmp_path / "hello.txt").write_text("cwd works")
    result = tool.run("cat hello.txt", cwd=str(tmp_path))
    assert result["exit_code"] == 0
    assert "cwd works" in result["stdout"]


@pytest.mark.skipif(
    __import__("shutil").which("npm") is None,
    reason="npm not installed"
)
def test_run_npm_style_command_in_project(tmp_path):
    # Simulate running a project command (package.json with a script)
    (tmp_path / "package.json").write_text('{"scripts": {"greet": "echo hello project"}}')
    result = ShellTool().run("npm run greet", cwd=str(tmp_path))
    assert result["exit_code"] == 0
    assert "hello project" in result["stdout"]


# ── find_files: mdfind ────────────────────────────────────────────────────────

def test_find_files_uses_mdfind_when_available(tmp_path):
    """find_files delegates to mdfind on macOS when mdfind is on PATH."""
    mock_result = MagicMock()
    mock_result.stdout = str(tmp_path / "hello.py") + "\n"
    mock_result.returncode = 0
    with patch("shutil.which", return_value="/usr/bin/mdfind"), \
         patch("tools.shell.subprocess.run", return_value=mock_result) as mock_run:
        result = ShellTool().find_files("*.py", directory=str(tmp_path))
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/mdfind"
    assert result["error"] is None
    assert result["count"] >= 1


def test_find_files_falls_back_to_rglob_when_mdfind_unavailable(tmp_path):
    """find_files falls back to rglob when mdfind is not on PATH."""
    (tmp_path / "test.py").write_text("")
    with patch("shutil.which", return_value=None):
        result = ShellTool().find_files("*.py", directory=str(tmp_path))
    assert result["error"] is None
    assert any("test.py" in m for m in result["matches"])


# ── search_content: ripgrep ───────────────────────────────────────────────────

def test_search_content_uses_ripgrep_when_available(tmp_path):
    """search_content uses rg when ripgrep is on PATH."""
    mock_result = MagicMock()
    mock_result.stdout = str(tmp_path / "foo.py") + "\n"
    mock_result.returncode = 0
    with patch("shutil.which", return_value="/usr/local/bin/rg"), \
         patch("tools.shell.subprocess.run", return_value=mock_result) as mock_run:
        ShellTool().search_content("pattern", directory=str(tmp_path))
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/local/bin/rg"


def test_search_content_falls_back_to_grep_when_rg_unavailable(tmp_path):
    """search_content falls back to grep when rg is not on PATH."""
    (tmp_path / "file.py").write_text("hello world\n")
    with patch("shutil.which", return_value=None):
        result = ShellTool().search_content("hello", directory=str(tmp_path))
    assert result["error"] is None


def test_search_content_timeout_is_at_least_30s(tmp_path):
    """search_content uses a timeout of at least 30 seconds."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.returncode = 0
    with patch("shutil.which", return_value=None), \
         patch("tools.shell.subprocess.run", return_value=mock_result) as mock_run:
        ShellTool().search_content("x", directory=str(tmp_path))
    timeout = mock_run.call_args.kwargs.get("timeout") or mock_run.call_args[1].get("timeout", 0)
    assert timeout >= 30
