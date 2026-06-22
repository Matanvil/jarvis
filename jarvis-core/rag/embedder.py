import httpx
from typing import List

MAX_EMBED_CHARS = 6000  # nomic-embed-text context limit ~8192 tokens; 6000 chars is safe


class EmbedderError(Exception):
    pass


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> List[float]:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text[:MAX_EMBED_CHARS]},
                )
                response.raise_for_status()
                data = response.json()
                if "embedding" not in data:
                    raise EmbedderError("Unexpected response from Ollama — is nomic-embed-text pulled?")
                return data["embedding"]
        except httpx.ConnectError:
            raise EmbedderError("Ollama is not running. Start it with: ollama serve")
        except httpx.TimeoutException:
            raise EmbedderError("Ollama timed out. Is nomic-embed-text loaded?")
        except httpx.HTTPStatusError as e:
            raise EmbedderError(f"Embedding failed: {e}")
