import pytest
from unittest.mock import MagicMock
from tools._dispatch import execute_tool
from guardrails import Guardrails


def make_guardrails():
    config = {"guardrails": {"read_files": "auto_allow"}}
    return Guardrails(config)


def test_search_codebase_dispatches_to_rag():
    rag = MagicMock()
    rag.search.return_value = {
        "chunks": [{"file": "a.py", "start_line": 1, "text": "def foo(): pass", "score": 0.9, "chunk_type": "function"}],
        "stale": False,
        "error": None,
    }
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "search_codebase",
        {"query": "authentication", "repo_path": "/my/repo"},
        shell, web, code, macos, make_guardrails(),
        rag=rag,
    )
    rag.search.assert_called_once_with("authentication", "/my/repo", n_results=5)
    assert "a.py:1" in result
    assert "def foo(): pass" in result


def test_search_codebase_includes_stale_warning():
    rag = MagicMock()
    rag.search.return_value = {
        "chunks": [{"file": "b.py", "start_line": 3, "text": "x = 1", "score": 0.7, "chunk_type": "block"}],
        "stale": True,
        "error": None,
    }
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "search_codebase",
        {"query": "config", "repo_path": "/my/repo"},
        shell, web, code, macos, make_guardrails(),
        rag=rag,
    )
    assert "stale" in result.lower()


def test_index_codebase_dispatches_to_rag():
    rag = MagicMock()
    rag.status.return_value = {"count": 0, "stale": True, "indexed_at": "never", "error": None}
    rag.index.return_value = {"indexed": 42, "error": None}
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "index_codebase",
        {"repo_path": "/my/repo"},
        shell, web, code, macos, make_guardrails(),
        rag=rag,
    )
    rag.index.assert_called_once_with("/my/repo")
    assert "42" in result


def test_index_codebase_skips_when_fresh():
    rag = MagicMock()
    rag.status.return_value = {"count": 100, "stale": False, "indexed_at": "2026-06-22T12:00:00+00:00", "error": None}
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "index_codebase",
        {"repo_path": "/my/repo"},
        shell, web, code, macos, make_guardrails(),
        rag=rag,
    )
    rag.index.assert_not_called()
    assert "up to date" in result
    assert "100" in result


def test_index_codebase_force_reindexes_when_fresh():
    rag = MagicMock()
    rag.status.return_value = {"count": 100, "stale": False, "indexed_at": "2026-06-22T12:00:00+00:00", "error": None}
    rag.index.return_value = {"indexed": 105, "error": None}
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "index_codebase",
        {"repo_path": "/my/repo", "force": True},
        shell, web, code, macos, make_guardrails(),
        rag=rag,
    )
    rag.index.assert_called_once_with("/my/repo")
    assert "105" in result


def test_index_codebase_falls_back_to_default_cwd():
    rag = MagicMock()
    rag.status.return_value = {"count": 0, "stale": True, "indexed_at": "never", "error": None}
    rag.index.return_value = {"indexed": 7, "error": None}
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "index_codebase",
        {},
        shell, web, code, macos, make_guardrails(),
        default_cwd="/default/project",
        rag=rag,
    )
    rag.index.assert_called_once_with("/default/project")


def test_search_codebase_no_rag_returns_error():
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "search_codebase",
        {"query": "foo", "repo_path": "/x"},
        shell, web, code, macos, make_guardrails(),
    )
    assert "error" in result.lower()


def test_index_codebase_no_repo_returns_error():
    rag = MagicMock()
    shell = MagicMock(); web = MagicMock(); code = MagicMock(); macos = MagicMock()
    result = execute_tool(
        "index_codebase",
        {},
        shell, web, code, macos, make_guardrails(),
        rag=rag,
    )
    assert "error" in result.lower()
