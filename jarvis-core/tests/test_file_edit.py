"""TDD tests for file_edit tool — written before implementation."""
import pytest
from pathlib import Path
from tools.shell import ShellTool
from tools._dispatch import execute_tool
from guardrails import Guardrails


@pytest.fixture
def shell():
    return ShellTool()


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "sample.swift"
    f.write_text('let x = 1\nlet y = 2\nlet z = 3\n')
    return f


@pytest.fixture
def guardrails():
    return Guardrails({"guardrails": {"edit_files": "auto_allow", "read_files": "auto_allow"}})


# --- ShellTool.file_edit unit tests ---

def test_file_edit_replaces_exact_string(shell, tmp_file):
    result = shell.file_edit(str(tmp_file), old_string="let y = 2", new_string="let y = 99")
    assert result["success"] is True
    assert "let y = 99" in tmp_file.read_text()
    assert "let y = 2" not in tmp_file.read_text()


def test_file_edit_returns_error_when_old_string_not_found(shell, tmp_file):
    result = shell.file_edit(str(tmp_file), old_string="let q = 9", new_string="anything")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_file_edit_returns_error_when_old_string_ambiguous(shell, tmp_file):
    tmp_file.write_text("foo\nfoo\nbar\n")
    result = shell.file_edit(str(tmp_file), old_string="foo", new_string="baz")
    assert result["success"] is False
    assert "2" in result["error"]   # reports how many occurrences found


def test_file_edit_replace_all_replaces_every_occurrence(shell, tmp_file):
    tmp_file.write_text("foo\nfoo\nbar\n")
    result = shell.file_edit(str(tmp_file), old_string="foo", new_string="baz", replace_all=True)
    assert result["success"] is True
    assert tmp_file.read_text() == "baz\nbaz\nbar\n"


def test_file_edit_returns_error_for_missing_file(shell, tmp_path):
    result = shell.file_edit(str(tmp_path / "nope.txt"), old_string="x", new_string="y")
    assert result["success"] is False
    assert "not found" in result["error"].lower() or "no such" in result["error"].lower()


def test_file_edit_preserves_rest_of_file(shell, tmp_file):
    result = shell.file_edit(str(tmp_file), old_string="let y = 2", new_string="let y = 42")
    assert result["success"] is True
    content = tmp_file.read_text()
    assert "let x = 1" in content
    assert "let z = 3" in content


# --- execute_tool integration ---

def test_execute_tool_file_edit(tmp_file, guardrails):
    shell = ShellTool()
    result = execute_tool(
        "file_edit",
        {"path": str(tmp_file), "old_string": "let x = 1", "new_string": "let x = 100"},
        shell, None, None, None, guardrails,
    )
    assert "success" in result.lower()
    assert "let x = 100" in tmp_file.read_text()


def test_execute_tool_file_edit_not_found(tmp_file, guardrails):
    shell = ShellTool()
    result = execute_tool(
        "file_edit",
        {"path": str(tmp_file), "old_string": "MISSING", "new_string": "x"},
        shell, None, None, None, guardrails,
    )
    assert "error" in result.lower() or "not found" in result.lower()
