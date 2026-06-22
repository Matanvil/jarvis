import pytest
from unittest.mock import MagicMock, patch
from tools.rag import RAGTool


@pytest.fixture
def mem(tmp_path):
    from memory import ProjectMemory
    return ProjectMemory(base_dir=tmp_path)


@pytest.fixture
def rag(mem):
    return RAGTool(memory=mem, ollama_host="http://localhost:11434")


def test_index_returns_chunk_count(rag, tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_store = MagicMock()
    mock_store.clear = MagicMock()
    mock_store.add = MagicMock()

    with patch("tools.rag.OllamaEmbedder", return_value=mock_embedder), \
         patch("tools.rag.VectorStore", return_value=mock_store), \
         patch("tools.rag.index_repo", return_value=5) as mock_index:
        result = rag.index(str(tmp_path))

    assert result["error"] is None
    assert result["indexed"] == 5
    mock_index.assert_called_once()


def test_index_saves_timestamp(rag, tmp_path, mem):
    with patch("tools.rag.OllamaEmbedder"), \
         patch("tools.rag.VectorStore"), \
         patch("tools.rag.index_repo", return_value=3):
        rag.index(str(tmp_path))

    data = mem.load(str(tmp_path))
    assert "rag_indexed_at" in data
    assert data["rag_repo_path"] == str(tmp_path)


def test_index_returns_error_on_exception(rag, tmp_path):
    with patch("tools.rag.OllamaEmbedder"), \
         patch("tools.rag.VectorStore"), \
         patch("tools.rag.index_repo", side_effect=RuntimeError("boom")):
        result = rag.index(str(tmp_path))
    assert result["error"] is not None
    assert result["indexed"] == 0


def test_search_returns_not_indexed_error_when_store_empty(rag, tmp_path):
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    with patch("tools.rag.OllamaEmbedder"), patch("tools.rag.VectorStore", return_value=mock_store):
        result = rag.search("authentication logic", str(tmp_path))
    assert result["error"] is not None
    assert "index_codebase" in result["error"]


def test_search_returns_chunks_with_metadata(rag, tmp_path, mem):
    from rag.models import Chunk
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.5, 0.5]
    mock_store = MagicMock()
    mock_store.count.return_value = 10
    mock_store.search.return_value = [
        Chunk(text="def auth(): ...", file="auth.py", start_line=5, score=0.92, chunk_type="function")
    ]
    mem.update_rag_index(str(tmp_path), str(tmp_path))

    with patch("tools.rag.OllamaEmbedder", return_value=mock_embedder), \
         patch("tools.rag.VectorStore", return_value=mock_store):
        result = rag.search("authentication", str(tmp_path), n_results=3)

    assert result["error"] is None
    assert result["stale"] is False
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["file"] == "auth.py"
    assert result["chunks"][0]["score"] == 0.92
    mock_store.search.assert_called_once_with([0.5, 0.5], n_results=3)


def test_search_marks_stale_when_index_is_old(rag, tmp_path, mem):
    from datetime import datetime, timezone, timedelta
    from rag.models import Chunk
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    mem.save(str(tmp_path), {"rag_indexed_at": old_ts})

    mock_store = MagicMock()
    mock_store.count.return_value = 5
    mock_store.search.return_value = [Chunk(text="x", file="f.py", start_line=1, score=0.8)]
    with patch("tools.rag.OllamaEmbedder") as MockEmb, \
         patch("tools.rag.VectorStore", return_value=mock_store):
        MockEmb.return_value.embed.return_value = [0.1]
        result = rag.search("query", str(tmp_path))

    assert result["stale"] is True
