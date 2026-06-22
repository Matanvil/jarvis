import pytest
import chromadb
from rag.store import VectorStore
from rag.models import Chunk


@pytest.fixture
def store():
    client = chromadb.EphemeralClient()
    return VectorStore(collection_name="test_col", _client=client)


def test_count_is_zero_on_empty_store(store):
    assert store.count() == 0


def test_add_and_count(store):
    chunks = [Chunk(text="def foo(): pass", file="a.py", start_line=1, chunk_type="function")]
    embeddings = [[0.1, 0.2, 0.3]]
    store.add(chunks, embeddings)
    assert store.count() == 1


def test_search_returns_chunks(store):
    chunks = [
        Chunk(text="def authenticate(user): ...", file="auth.py", start_line=10, chunk_type="function"),
        Chunk(text="def render_page(): ...", file="views.py", start_line=5, chunk_type="function"),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.add(chunks, embeddings)
    results = store.search([1.0, 0.0, 0.0], n_results=1)
    assert len(results) == 1
    assert results[0].file == "auth.py"
    assert results[0].score > 0.9


def test_search_on_empty_store_returns_empty(store):
    results = store.search([0.1, 0.2, 0.3], n_results=5)
    assert results == []


def test_keyword_search_returns_matching_chunks(store):
    chunks = [
        Chunk(text="def fetch_user(id): return db.get(id)", file="db.py", start_line=1),
        Chunk(text="class Config: pass", file="config.py", start_line=1),
    ]
    store.add(chunks, [[0.1, 0.0], [0.0, 0.1]])
    results = store.keyword_search("fetch_user")
    assert len(results) == 1
    assert results[0].file == "db.py"


def test_clear_resets_store(store):
    store.add(
        [Chunk(text="x = 1", file="f.py", start_line=1)],
        [[0.5, 0.5]],
    )
    store.clear()
    assert store.count() == 0
