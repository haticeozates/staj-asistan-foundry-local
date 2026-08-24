"""Vector storage.

The default store is an exact-cosine NumPy index: for a corpus of a few thousand
chunks an exhaustive search is both faster and more predictable than an ANN
index, it has zero extra dependencies, and it is trivially testable. A ChromaDB
backend is available behind ``pip install '.[chroma]'`` for anyone who wants
persistent, larger-scale storage; both satisfy the same protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .models import AuthorRole, Chunk, ChunkCategory
from .topics import classify_chunk


@runtime_checkable
class VectorStore(Protocol):
    """Everything the retriever needs from a store."""

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


def _chunk_to_dict(chunk: Chunk) -> dict:
    data = asdict(chunk)
    data["role"] = chunk.role.value
    data["category"] = chunk.category.value
    data["senders"] = list(chunk.senders)
    data["start_time"] = chunk.start_time.isoformat() if chunk.start_time else None
    data["end_time"] = chunk.end_time.isoformat() if chunk.end_time else None
    return data


def _chunk_from_dict(data: dict) -> Chunk:
    text = data["text"]
    source = data["source"]
    raw_category = data.get("category")
    if raw_category:
        category = ChunkCategory(raw_category)
    else:
        category = classify_chunk(text, source)
    return Chunk(
        chunk_id=data["chunk_id"],
        text=text,
        source=source,
        role=AuthorRole(data["role"]),
        senders=tuple(data.get("senders", ())),
        start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
        end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
        message_count=data.get("message_count", 1),
        instructor_ratio=data.get("instructor_ratio", 0.0),
        category=category,
    )


class NumpyVectorStore:
    """Exact cosine-similarity store backed by a single NumPy matrix."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )
        if not chunks:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._matrix is None:
            self._matrix = vectors
        else:
            if self._matrix.shape[1] != vectors.shape[1]:
                raise ValueError(
                    "embedding dimension changed; rebuild the index after switching backend"
                )
            self._matrix = np.vstack([self._matrix, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks or top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm
        scores = self._matrix @ query
        limit = min(top_k, len(self._chunks))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def vectors(self) -> np.ndarray | None:
        return self._matrix

    def clear(self) -> None:
        self._chunks = []
        self._matrix = None

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, directory: str | Path) -> None:
        """Persist the index locally. Never commit the resulting directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.save(path / "vectors.npy", self._matrix)
        (path / "chunks.json").write_text(
            json.dumps([_chunk_to_dict(c) for c in self._chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> NumpyVectorStore:
        path = Path(directory)
        store = cls()
        chunks_file = path / "chunks.json"
        if not chunks_file.exists():
            return store
        chunks = [_chunk_from_dict(d) for d in json.loads(chunks_file.read_text(encoding="utf-8"))]
        vectors_file = path / "vectors.npy"
        vectors = (
            np.load(vectors_file)
            if vectors_file.exists()
            else np.zeros((len(chunks), 1), dtype=np.float32)
        )
        store.add(chunks, vectors)
        return store


class ChromaVectorStore:
    """Optional persistent backend. Same protocol, different storage engine."""

    def __init__(self, collection_name: str = "staj_asistan", persist_directory: str | None = None):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaDB backend requires: pip install '.[chroma]'"
            ) from exc
        self._client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.EphemeralClient()
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._collection_name = collection_name

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[v.tolist() for v in np.asarray(vectors, dtype=np.float32)],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "role": c.role.value,
                    "senders": "|".join(c.senders),
                    "start_time": c.start_time.isoformat() if c.start_time else "",
                    "end_time": c.end_time.isoformat() if c.end_time else "",
                    "message_count": c.message_count,
                    "instructor_ratio": c.instructor_ratio,
                    "category": c.category.value,
                }
                for c in chunks
            ],
        )

    def _to_chunk(self, chunk_id: str, document: str, metadata: dict) -> Chunk:
        return _chunk_from_dict(
            {
                "chunk_id": chunk_id,
                "text": document,
                "source": metadata.get("source", "?"),
                "role": metadata.get("role", AuthorRole.PARTICIPANT.value),
                "senders": [s for s in metadata.get("senders", "").split("|") if s],
                "start_time": metadata.get("start_time") or None,
                "end_time": metadata.get("end_time") or None,
                "message_count": metadata.get("message_count", 1),
                "instructor_ratio": metadata.get("instructor_ratio", 0.0),
                "category": metadata.get("category"),
            }
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if len(self) == 0 or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=min(top_k, len(self)),
        )
        out: list[tuple[Chunk, float]] = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            out.append((self._to_chunk(chunk_id, document, metadata), 1.0 - float(distance)))
        return out

    def all_chunks(self) -> list[Chunk]:
        data = self._collection.get()
        return [
            self._to_chunk(i, d, m)
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def __len__(self) -> int:
        return int(self._collection.count())


def build_vector_store(backend: str = "numpy", **kwargs) -> VectorStore:
    """Factory used by the pipeline and the UI."""
    backend = backend.strip().lower()
    if backend in {"numpy", "memory", "default"}:
        return NumpyVectorStore()
    if backend == "chroma":
        return ChromaVectorStore(**kwargs)
    raise ValueError(f"Unknown vector store backend '{backend}' (use 'numpy' or 'chroma')")
