import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from memory import ProjectMemory


@pytest.fixture
def mem(tmp_path):
    return ProjectMemory(base_dir=tmp_path)


def test_load_returns_empty_dict_for_unknown_project(mem):
    result = mem.load("/some/new/project")
    assert result == {}


def test_save_and_load_roundtrip(mem):
    data = {"build_command": "npm run build", "test_command": "npm test"}
    mem.save("/my/project", data)
    loaded = mem.load("/my/project")
    assert loaded["build_command"] == "npm run build"
    assert loaded["test_command"] == "npm test"


def test_update_merges_with_existing(mem):
    mem.save("/my/project", {"build_command": "npm run build", "project_type": "node"})
    mem.update("/my/project", "test_command", "npm test")
    loaded = mem.load("/my/project")
    assert loaded["build_command"] == "npm run build"   # preserved
    assert loaded["test_command"] == "npm test"         # added


def test_different_projects_stored_separately(mem):
    mem.save("/project/a", {"build_command": "make"})
    mem.save("/project/b", {"build_command": "cargo build"})
    assert mem.load("/project/a")["build_command"] == "make"
    assert mem.load("/project/b")["build_command"] == "cargo build"


def test_format_context_returns_empty_string_for_unknown_project(mem):
    result = mem.format_context("/unknown/project")
    assert result == ""


def test_format_context_returns_readable_string(mem):
    mem.save("/my/project", {
        "project_type": "node",
        "build_command": "npm run build",
        "test_command": "npm test",
        "notes": "Uses Vite",
    })
    ctx = mem.format_context("/my/project")
    assert "npm run build" in ctx
    assert "npm test" in ctx
    assert "Vite" in ctx


def test_discover_with_ollama_saves_memory(mem, tmp_path):
    """discover() calls Ollama and saves the result."""
    project_dir = tmp_path / "myapp"
    project_dir.mkdir()
    (project_dir / "package.json").write_text(
        json.dumps({"scripts": {"build": "vite build", "test": "vitest"}})
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "project_type": "node",
            "build_command": "npm run build",
            "test_command": "npm test",
        })}}]
    }
    with patch("httpx.Client.post", return_value=mock_response):
        result = mem.discover(str(project_dir), ollama_host="http://localhost:11434", ollama_model="mistral:latest")
    assert result["project_type"] == "node"
    saved = mem.load(str(project_dir))
    assert saved["build_command"] == "npm run build"


def test_discover_returns_empty_on_ollama_failure(mem, tmp_path):
    """discover() returns {} gracefully if Ollama is unreachable."""
    import httpx
    project_dir = tmp_path / "myapp"
    project_dir.mkdir()
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("refused")):
        result = mem.discover(str(project_dir), ollama_host="http://localhost:11434", ollama_model="mistral:latest")
    assert result == {}
