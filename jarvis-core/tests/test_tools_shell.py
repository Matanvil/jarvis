import pytest
from pathlib import Path
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
