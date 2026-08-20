"""Embedding backends, resolved in a documented order.

The project targets **Microsoft Foundry Local** first: if it exposes an
embedding model over its OpenAI-compatible endpoint, that is what gets used.
Foundry Local's catalog is primarily chat models though, so two fallbacks keep
the application usable and the test suite hermetic:

1. ``foundry`` - Foundry Local ``/v1/embeddings`` (primary)
2. ``sentence-transformers`` - local multilingual model, still fully offline
3. ``hashing`` - dependency-free deterministic vectoriser (tests, CI, cold start)

Whichever is active is surfaced in the UI and in the README, so a reader always
knows which backend produced a given result.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

DEFAULT_FOUNDRY_ENDPOINT = "http://localhost:5273/v1"
DEFAULT_FOUNDRY_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class EmbeddingBackendUnavailable(RuntimeError):
    """Raised when a backend cannot be initialised, so the chain can fall through."""


class EmbeddingBackend(ABC):
    """Minimal interface every backend implements."""

    name: str = "unknown"
    dimension: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, dimension)`` float32 matrix of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @property
    def description(self) -> str:
        return f"{self.name} (dim={self.dimension})"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def turkish_tokens(text: str) -> list[str]:
    """Lowercase with correct Turkish dotted/dotless ``i`` handling."""
    lowered = text.replace("I", "ı").replace("İ", "i").lower()
    return _TOKEN_RE.findall(lowered)


class HashingEmbeddings(EmbeddingBackend):
    """Deterministic hashing vectoriser: no downloads, no network, no state.

    Combines word unigrams/bigrams with character 4-grams, which keeps it usable
    for a morphologically rich language like Turkish where suffixes change words
    but stems stay stable. It is a floor, not a ceiling - the real quality comes
    from the Foundry Local or sentence-transformers backends.
    """

    name = "hashing"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _features(self, text: str) -> dict[str, float]:
        tokens = turkish_tokens(text)
        counts: dict[str, float] = {}
        for token in tokens:
            counts[f"w:{token}"] = counts.get(f"w:{token}", 0.0) + 1.0
            for size in (4, 5):
                for i in range(max(len(token) - size + 1, 0)):
                    gram = f"c:{token[i : i + size]}"
                    counts[gram] = counts.get(gram, 0.0) + 0.5
        for first, second in zip(tokens, tokens[1:]):
            key = f"b:{first}_{second}"
            counts[key] = counts.get(key, 0.0) + 0.75
        return counts

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text).items():
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, bucket] += sign * (1.0 + math.log(weight + 1.0))
        return _l2_normalize(matrix)


class FoundryLocalEmbeddings(EmbeddingBackend):
    """Embeddings from Foundry Local's OpenAI-compatible ``/v1/embeddings``."""

    name = "foundry-local"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        import requests

        self._requests = requests
        self.endpoint = (endpoint or os.getenv("FOUNDRY_LOCAL_ENDPOINT", DEFAULT_FOUNDRY_ENDPOINT)).rstrip("/")
        self.model = model or os.getenv(
            "FOUNDRY_LOCAL_EMBEDDING_MODEL", DEFAULT_FOUNDRY_EMBEDDING_MODEL
        )
        self.timeout = timeout
        self.dimension = self._probe()

    def _probe(self) -> int:
        try:
            vector = self._call(["merhaba"])[0]
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure means fall through
            raise EmbeddingBackendUnavailable(
                f"Foundry Local embedding endpoint unreachable at {self.endpoint}: {exc}"
            ) from exc
        return len(vector)

    def _call(self, texts: list[str]) -> list[list[float]]:
        response = self._requests.post(
            f"{self.endpoint}/embeddings",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._call(texts[start : start + batch_size]))
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    @property
    def description(self) -> str:
        return f"{self.name}:{self.model} (dim={self.dimension})"


class SentenceTransformerEmbeddings(EmbeddingBackend):
    """Local multilingual sentence-transformers model - runs fully offline once cached."""

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "STAJ_ASISTAN_ST_MODEL", DEFAULT_SENTENCE_TRANSFORMER
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingBackendUnavailable(
                "sentence-transformers is not installed (pip install '.[embeddings]')"
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # noqa: BLE001 - model download/load failure
            raise EmbeddingBackendUnavailable(
                f"sentence-transformers model '{self.model_name}' could not be loaded: {exc}"
            ) from exc
        get_dimension = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    @property
    def description(self) -> str:
        return f"{self.name}:{self.model_name.split('/')[-1]} (dim={self.dimension})"


#: Order used when the backend is set to ``auto``.
FALLBACK_ORDER = ("foundry", "sentence-transformers", "hashing")

_BUILDERS = {
    "foundry": FoundryLocalEmbeddings,
    "foundry-local": FoundryLocalEmbeddings,
    "sentence-transformers": SentenceTransformerEmbeddings,
    "hashing": HashingEmbeddings,
}


def resolve_embedding_backend(preference: str | None = None) -> EmbeddingBackend:
    """Build an embedding backend, honouring the preference and falling back safely.

    Preference resolution order: explicit argument, ``STAJ_ASISTAN_EMBEDDING_BACKEND``
    environment variable, then ``auto``. Setting ``STAJ_ASISTAN_OFFLINE=1`` skips
    every backend that may touch the network.
    """
    choice = (preference or os.getenv("STAJ_ASISTAN_EMBEDDING_BACKEND", "auto")).strip().lower()
    offline = os.getenv("STAJ_ASISTAN_OFFLINE", "").strip() in {"1", "true", "yes"}

    if choice != "auto":
        builder = _BUILDERS.get(choice)
        if builder is None:
            raise ValueError(
                f"Unknown embedding backend '{choice}'. Valid: {sorted(set(_BUILDERS))} or 'auto'."
            )
        return builder()

    for candidate in FALLBACK_ORDER:
        if offline and candidate != "hashing":
            continue
        try:
            return _BUILDERS[candidate]()
        except EmbeddingBackendUnavailable:
            continue
    return HashingEmbeddings()


@lru_cache(maxsize=4)
def cached_backend(preference: str | None = None) -> EmbeddingBackend:
    """Cache backends so Streamlit reruns do not reload a model every time."""
    return resolve_embedding_backend(preference)
