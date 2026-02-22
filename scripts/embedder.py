"""Ollama embedding client with LRU cache."""
import collections
import hashlib
import logging
import requests

log = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text", cache_size: int = 256) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._cache: collections.OrderedDict[str, list[float]] = collections.OrderedDict()
        self._cache_size = cache_size

    def embed(self, text: str) -> list[float]:
        """Return 768-dim embedding. Uses LRU cache. Returns [] on failure."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=30,
            )
            resp.raise_for_status()
            embedding = resp.json()["embeddings"][0]
        except Exception as e:
            log.warning("Embedding failed: %s", e)
            return []

        # LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially."""
        return [self.embed(t) for t in texts]

    def dims(self) -> int:
        return 768

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
