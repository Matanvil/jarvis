from rag.embedder import OllamaEmbedder
from rag.indexer import index_repo
from rag.store import VectorStore


class RAGTool:
    def __init__(self, memory, ollama_host: str = "http://localhost:11434"):
        self._memory = memory
        self._ollama_host = ollama_host

    def _chroma_path(self, repo_path: str) -> str:
        return str(self._memory.rag_project_dir(repo_path) / "rag_store")

    def status(self, repo_path: str) -> dict:
        """Return current index state without modifying anything."""
        try:
            store = VectorStore(chroma_path=self._chroma_path(repo_path))
            count = store.count()
            stale = self._memory.rag_is_stale(repo_path)
            data = self._memory.load(repo_path)
            return {"count": count, "stale": stale, "indexed_at": data.get("rag_indexed_at", "never"), "error": None}
        except Exception as e:
            return {"count": 0, "stale": True, "indexed_at": "never", "error": str(e)}

    def index(self, repo_path: str) -> dict:
        try:
            embedder = OllamaEmbedder(base_url=self._ollama_host)
            store = VectorStore(chroma_path=self._chroma_path(repo_path))
            count = index_repo(repo_path, embedder, store, use_semantic=True)
            self._memory.update_rag_index(repo_path, repo_path)
            return {"indexed": count, "error": None}
        except Exception as e:
            return {"indexed": 0, "error": str(e)}

    def search(self, query: str, repo_path: str, n_results: int = 5) -> dict:
        try:
            embedder = OllamaEmbedder(base_url=self._ollama_host)
            store = VectorStore(chroma_path=self._chroma_path(repo_path))
            if store.count() == 0:
                return {"chunks": [], "stale": False, "error": "Codebase not indexed. Run index_codebase first."}
            stale = self._memory.rag_is_stale(repo_path)
            embedding = embedder.embed(query)
            chunks = store.search(embedding, n_results=n_results)
            return {
                "chunks": [
                    {
                        "file": c.file,
                        "start_line": c.start_line,
                        "text": c.text,
                        "score": c.score,
                        "chunk_type": c.chunk_type,
                    }
                    for c in chunks
                ],
                "stale": stale,
                "error": None,
            }
        except Exception as e:
            return {"chunks": [], "stale": False, "error": str(e)}
