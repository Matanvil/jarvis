import pytest
from unittest.mock import MagicMock
from tools._dispatch import execute_tool, TOOL_TO_GUARDRAIL_CATEGORY
from tools._errors import ApprovalRequiredError
from guardrails import Guardrails


def make_guardrails(overrides=None):
    config = {
        "guardrails": {
            "read_files": "auto_allow",
            **(overrides or {}),
        }
    }
    return Guardrails(config)


def base_args(tool_name, tool_input):
    shell = MagicMock()
    web = MagicMock()
    code = MagicMock()
    macos = MagicMock()
    guardrails = make_guardrails()
    return (tool_name, tool_input, shell, web, code, macos, guardrails)


# --- Guardrail category tests ---

def test_coding_ask_maps_to_read_files():
    assert TOOL_TO_GUARDRAIL_CATEGORY["coding_ask"] == "read_files"


def test_coding_plan_maps_to_read_files():
    assert TOOL_TO_GUARDRAIL_CATEGORY["coding_plan"] == "read_files"


def test_coding_review_maps_to_read_files():
    assert TOOL_TO_GUARDRAIL_CATEGORY["coding_review"] == "read_files"


# --- Dispatch routing tests ---

def test_coding_ask_dispatch_calls_coding_ask():
    """execute_tool routes coding_ask to coding.ask() with correct params."""
    coding = MagicMock()
    coding.ask.return_value = {"answer": "the router classifies intent", "error": None}
    args = base_args("coding_ask", {"question": "what does the router do?", "cwd": "/tmp/proj"})
    result = execute_tool(*args, coding=coding)
    coding.ask.assert_called_once_with("what does the router do?", "/tmp/proj")
    assert result == "the router classifies intent"


def test_coding_plan_dispatch_calls_coding_plan():
    """execute_tool routes coding_plan to coding.plan() with correct params."""
    coding = MagicMock()
    coding.plan.return_value = {
        "plan_summary": "Plan: refactor auth\n\n• auth.py: Extract service",
        "edits": [
            {"file": "auth.py", "description": "Extract service", "old_code": "old", "new_code": "new"}
        ],
        "error": None,
    }
    args = base_args("coding_plan", {"task": "refactor auth", "cwd": "/tmp/proj"})
    result = execute_tool(*args, coding=coding)
    coding.plan.assert_called_once_with("refactor auth", "/tmp/proj")
    assert "auth.py" in result
    assert "Extract service" in result
    assert "--- old ---" in result
    assert "--- new ---" in result


def test_coding_review_dispatch_calls_coding_review():
    """execute_tool routes coding_review to coding.review() with correct params."""
    coding = MagicMock()
    coding.review.return_value = {
        "summary": "1 critical issue found.",
        "issues": [
            {"category": "critical", "description": "SQL injection", "file": "db.py", "recommendation": "Use params"}
        ],
        "error": None,
    }
    args = base_args("coding_review", {"cwd": "/tmp/proj", "context": "db changes"})
    result = execute_tool(*args, coding=coding)
    coding.review.assert_called_once_with("/tmp/proj", "db changes")
    assert "[CRITICAL]" in result
    assert "SQL injection" in result
    assert "→ Use params" in result


def test_coding_review_context_defaults_to_empty_string():
    """execute_tool passes empty string for context when not provided."""
    coding = MagicMock()
    coding.review.return_value = {"summary": "ok", "issues": [], "error": None}
    args = base_args("coding_review", {"cwd": "/tmp/proj"})
    execute_tool(*args, coding=coding)
    coding.review.assert_called_once_with("/tmp/proj", "")


# --- Error propagation tests ---

def test_coding_ask_returns_error_string_on_tool_error():
    """execute_tool returns error string when coding.ask returns error dict."""
    coding = MagicMock()
    coding.ask.return_value = {"answer": None, "error": "Ollama is not running"}
    args = base_args("coding_ask", {"question": "q", "cwd": "/tmp"})
    result = execute_tool(*args, coding=coding)
    assert "Ollama is not running" in result


def test_coding_ask_unavailable_when_coding_is_none():
    """execute_tool returns error string when coding param is None."""
    args = base_args("coding_ask", {"question": "q", "cwd": "/tmp"})
    result = execute_tool(*args, coding=None)
    assert "coding agent not available" in result


# --- Guardrail blocking test ---

def test_coding_tools_blocked_when_read_files_requires_approval():
    """All three coding tools raise ApprovalRequiredError when read_files is require_approval."""
    coding = MagicMock()
    shell, web, code, macos = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    guardrails = make_guardrails({"read_files": "require_approval"})
    for tool_name, tool_input in [
        ("coding_ask", {"question": "q", "cwd": "/tmp"}),
        ("coding_plan", {"task": "t", "cwd": "/tmp"}),
        ("coding_review", {"cwd": "/tmp"}),
    ]:
        with pytest.raises(ApprovalRequiredError):
            execute_tool(tool_name, tool_input, shell, web, code, macos, guardrails, coding=coding)
