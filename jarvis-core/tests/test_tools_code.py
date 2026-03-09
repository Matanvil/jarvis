import pytest
from pathlib import Path
from tools.code import CodeTool


def test_run_python_snippet():
    tool = CodeTool()
    result = tool.run_snippet('print("hello from jarvis")', "python")
    assert result["exit_code"] == 0
    assert "hello from jarvis" in result["stdout"]


def test_run_python_with_error():
    tool = CodeTool()
    result = tool.run_snippet("raise ValueError('test error')", "python")
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]


def test_run_shell_script():
    tool = CodeTool()
    result = tool.run_snippet("echo 'from shell script'", "bash")
    assert result["exit_code"] == 0
    assert "from shell script" in result["stdout"]


def test_run_javascript_snippet():
    tool = CodeTool()
    result = tool.run_snippet('console.log("hello from node")', "javascript")
    assert result["exit_code"] == 0
    assert "hello from node" in result["stdout"]


def test_run_js_alias():
    tool = CodeTool()
    result = tool.run_snippet('console.log("js alias works")', "js")
    assert result["exit_code"] == 0
    assert "js alias works" in result["stdout"]


def test_run_snippet_with_cwd(tmp_path):
    # Write a file into tmp_path, then read it from a JS snippet using cwd
    (tmp_path / "data.txt").write_text("project data")
    tool = CodeTool()
    result = tool.run_snippet(
        "const fs = require('fs'); console.log(fs.readFileSync('data.txt', 'utf8'))",
        "javascript",
        cwd=str(tmp_path),
    )
    assert result["exit_code"] == 0
    assert "project data" in result["stdout"]


def test_run_python_with_cwd(tmp_path):
    (tmp_path / "msg.txt").write_text("from project")
    tool = CodeTool()
    result = tool.run_snippet(
        "print(open('msg.txt').read())",
        "python",
        cwd=str(tmp_path),
    )
    assert result["exit_code"] == 0
    assert "from project" in result["stdout"]


def test_unsupported_language_returns_error():
    tool = CodeTool()
    result = tool.run_snippet("code", "cobol")
    assert result["exit_code"] == -1
    assert "Unsupported language" in result["error"]


def test_convenience_shorthands():
    tool = CodeTool()
    assert tool.run_python('print("ok")')["exit_code"] == 0
    assert tool.run_js('console.log("ok")')["exit_code"] == 0
    assert tool.run_shell_script('echo ok')["exit_code"] == 0
