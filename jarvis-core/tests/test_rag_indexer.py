import json
import pytest
from pathlib import Path
from rag.indexer import scan_files, chunk_file_naive, chunk_file_semantic, index_repo
from rag.models import Chunk


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    return 'hi'\n\ndef world():\n    return 'world'\n"
    )
    (tmp_path / "README.md").write_text("# My Project\n\nDoes stuff.\n\n## Usage\n\nRun it.\n")
    (tmp_path / "config.json").write_text(json.dumps({"name": "app", "version": "1.0", "scripts": {"build": "make"}}))
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("module.exports = {}")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    return tmp_path


def test_scan_files_finds_py_md_json(repo):
    files = list(scan_files(str(repo)))
    names = {f.name for f in files}
    assert "main.py" in names
    assert "README.md" in names
    assert "config.json" in names


def test_scan_files_skips_node_modules_and_git(repo):
    files = list(scan_files(str(repo)))
    parts_sets = [set(f.parts) for f in files]
    assert not any("node_modules" in p for p in parts_sets)
    assert not any(".git" in p for p in parts_sets)


def test_chunk_file_naive_splits_by_lines(tmp_path):
    py_file = tmp_path / "big.py"
    py_file.write_text("\n".join(f"line {i}" for i in range(120)))
    chunks = chunk_file_naive(py_file, str(tmp_path), chunk_size=50)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].start_line == 1
    assert chunks[1].start_line == 51
    assert chunks[2].start_line == 101


def test_chunk_file_semantic_python_splits_by_function(tmp_path):
    py_file = tmp_path / "module.py"
    py_file.write_text(
        "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n\n\nclass Baz:\n    pass\n"
    )
    chunks = chunk_file_semantic(py_file, str(tmp_path))
    types = {c.chunk_type for c in chunks}
    first_lines = [c.text.splitlines()[0] for c in chunks]
    assert "function" in types
    assert "class" in types
    assert any("foo" in l for l in first_lines)
    assert any("bar" in l for l in first_lines)
    assert any("Baz" in l for l in first_lines)


def test_chunk_file_semantic_javascript_splits_by_function(tmp_path):
    js_file = tmp_path / "utils.js"
    js_file.write_text(
        "function fetchUser(id) {\n  return db.get(id);\n}\n\n"
        "class UserService {\n  constructor() {}\n}\n"
    )
    chunks = chunk_file_semantic(js_file, str(tmp_path))
    assert len(chunks) >= 1
    texts = " ".join(c.text for c in chunks)
    assert "fetchUser" in texts or "UserService" in texts


def test_chunk_file_semantic_typescript_splits_by_function(tmp_path):
    ts_file = tmp_path / "api.ts"
    ts_file.write_text(
        "interface User {\n  id: number;\n  name: string;\n}\n\n"
        "function getUser(id: number): User {\n  return users[id];\n}\n"
    )
    chunks = chunk_file_semantic(ts_file, str(tmp_path))
    assert len(chunks) >= 1
    texts = " ".join(c.text for c in chunks)
    assert "getUser" in texts or "User" in texts


def test_chunk_file_semantic_swift_splits_by_function(tmp_path):
    swift_file = tmp_path / "Controller.swift"
    swift_file.write_text(
        "class AudioController {\n    func startRecording() {}\n}\n\n"
        "struct Config {\n    var host: String\n}\n"
    )
    chunks = chunk_file_semantic(swift_file, str(tmp_path))
    assert len(chunks) >= 1
    texts = " ".join(c.text for c in chunks)
    assert "AudioController" in texts or "Config" in texts


def test_chunk_file_semantic_markdown_splits_by_header(tmp_path):
    md_file = tmp_path / "README.md"
    md_file.write_text(
        "# Project\n\nIntro paragraph.\n\n"
        "## Installation\n\nRun `pip install`.\n\n"
        "## Usage\n\nCall the CLI.\n"
    )
    chunks = chunk_file_semantic(md_file, str(tmp_path))
    assert len(chunks) == 3
    assert "Installation" in chunks[1].text
    assert "Usage" in chunks[2].text


def test_chunk_file_semantic_json_splits_by_top_level_key(tmp_path):
    json_file = tmp_path / "package.json"
    json_file.write_text(json.dumps({"name": "myapp", "version": "1.0", "scripts": {"build": "make"}}))
    chunks = chunk_file_semantic(json_file, str(tmp_path))
    assert len(chunks) == 3
    keys = [list(json.loads(c.text).keys())[0] for c in chunks]
    assert "name" in keys
    assert "version" in keys
    assert "scripts" in keys


def test_chunk_file_semantic_json_falls_back_for_arrays(tmp_path):
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([{"id": i} for i in range(60)]))
    chunks = chunk_file_semantic(json_file, str(tmp_path))
    assert len(chunks) >= 1


def test_chunk_file_semantic_falls_back_for_unparseable_python(tmp_path):
    py_file = tmp_path / "broken.py"
    py_file.write_text("def (this is not valid python")
    chunks = chunk_file_semantic(py_file, str(tmp_path))
    assert len(chunks) >= 1


def test_index_repo_calls_embedder_and_store(tmp_path):
    (tmp_path / "a.py").write_text("def greet():\n    return 'hi'\n")

    embedded = []
    stored = []

    class FakeEmbedder:
        def embed(self, text):
            embedded.append(text)
            return [0.1, 0.2]

    class FakeStore:
        def clear(self): pass
        def add(self, chunks, embeddings):
            stored.extend(chunks)

    count = index_repo(str(tmp_path), FakeEmbedder(), FakeStore(), use_semantic=True)
    assert count == len(stored)
    assert count > 0
    assert len(embedded) == count
