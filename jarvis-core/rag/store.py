import chromadb
from chromadb.errors import NotFoundError
from typing import List
from rag.models import Chunk


class VectorStore:
    def __init__(self, chroma_path: str = ".chroma", collection_name: str = "codebase", _client=None):
        self.path = chroma_path
        self.collection_name = collection_name
        self._client = _client
        self._collection = None
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except (ValueError, NotFoundError):
                pass

    def _get_collection(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.path)
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        collection = self._get_collection()
        ids = [f"{c.file}:{c.start_line}" for c in chunks]
        collection.add(
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            ids=ids,
            metadatas=[
                {"file": c.file, "start_line": c.start_line, "chunk_type": c.chunk_type}
                for c in chunks
            ],
        )

    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Chunk]:
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = round(1.0 - (dist / 2.0), 4)
            chunks.append(Chunk(
                text=doc,
                file=meta["file"],
                start_line=meta["start_line"],
                score=score,
                chunk_type=meta["chunk_type"],
            ))
        return chunks

    def keyword_search(self, keyword: str, n_results: int = 10) -> List[Chunk]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(
            where_document={"$contains": keyword},
            include=["documents", "metadatas"],
            limit=n_results,
        )
        return [
            Chunk(text=doc, file=meta["file"], start_line=meta["start_line"], chunk_type=meta["chunk_type"])
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    def clear(self) -> None:
        self._get_collection()
        try:
            self._client.delete_collection(self.collection_name)
        except (ValueError, NotFoundError):
            pass
        self._collection = None

    def count(self) -> int:
        return self._get_collection().count()
