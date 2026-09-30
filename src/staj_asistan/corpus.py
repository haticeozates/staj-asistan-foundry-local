"""Build and load persistent, privacy-scanned local corpora."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4

from .embeddings import EmbeddingBackend, resolve_embedding_backend
from .generation import ExtractiveClient, LLMClient
from .ingestion import IngestedSource, ingest_bytes
from .models import IndexStats
from .pipeline import Assistant
from .privacy import contains_personal_data
from .private_config import PrivateConfig
from .vector_store import NumpyVectorStore
from .whatsapp_parser import InstructorIdentity


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: int
    embedding_backend: str
    embedding_description: str
    embedding_dimension: int
    source_count: int
    message_count: int
    instructor_message_count: int
    chunk_count: int
    masked_entity_count: int
    source_labels: tuple[str, ...]
    input_hashes: tuple[str, ...]
    privacy_scan_clean: bool

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_labels"] = tuple(payload["source_labels"])
            payload["input_hashes"] = tuple(payload["input_hashes"])
            manifest = cls(**payload)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("corpus manifest is invalid") from exc
        if manifest.schema_version != 1:
            raise ValueError("corpus manifest schema is unsupported")
        return manifest


def _safe_label(config: PrivateConfig, input_path: Path, position: int) -> str:
    label = config.source_labels.get(input_path.name, f"Kaynak {position}")
    folded_label = label.casefold()
    if contains_personal_data(label) or any(
        alias.casefold() in folded_label for alias in config.instructor_aliases
    ):
        raise ValueError("source label failed the privacy scan")
    return label


def _validated_output_dir(output_dir: Path) -> Path:
    index_root = Path("data/index").resolve()
    resolved = Path(output_dir).resolve()
    try:
        relative = resolved.relative_to(index_root)
    except ValueError as exc:
        raise ValueError("corpus output must be under data/index") from exc
    if not relative.parts:
        raise ValueError("corpus output must be below data/index")
    return resolved


_INSTRUCTOR_ROLE_LABEL = "Eğitmen"


def _redact_aliases(text: str, aliases: tuple[str, ...]) -> str:
    """Replace configured instructor names so they cannot survive in the index."""
    redacted = text
    for alias in sorted((item.strip() for item in aliases if item.strip()), key=len, reverse=True):
        if len(alias) < 3:
            continue
        redacted = re.compile(re.escape(alias), re.IGNORECASE).sub(_INSTRUCTOR_ROLE_LABEL, redacted)
    return redacted


def _sanitise_sources(
    sources: list[IngestedSource],
    base_label: str,
    aliases: tuple[str, ...],
) -> list[IngestedSource]:
    sanitised: list[IngestedSource] = []
    for index, source in enumerate(sources, start=1):
        label = base_label if len(sources) == 1 else f"{base_label} {index}"
        source.source = label
        source.messages = [
            replace(
                message,
                text=_redact_aliases(message.text, aliases),
                source=label,
                sender=_INSTRUCTOR_ROLE_LABEL if message.is_instructor else message.sender,
            )
            for message in source.messages
        ]
        sanitised.append(source)
    return sanitised


def _chunks_are_private(assistant: Assistant, aliases: tuple[str, ...]) -> bool:
    folded_aliases = tuple(alias.casefold() for alias in aliases)
    for chunk in assistant.store.all_chunks():
        searchable = "\n".join((chunk.text, chunk.source, *chunk.senders))
        folded = searchable.casefold()
        if contains_personal_data(searchable) or any(alias in folded for alias in folded_aliases):
            return False
    return True


def _write_index_atomically(
    output_dir: Path,
    store: NumpyVectorStore,
    manifest: CorpusManifest,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent)))
    backup: Path | None = None
    try:
        store.save(temporary)
        (temporary / "manifest.json").write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid4().hex}")
            output_dir.replace(backup)
        temporary.replace(output_dir)
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError:
                pass
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_private_index(
    config: PrivateConfig,
    inputs: list[Path],
    output_dir: Path,
    embedding_backend: EmbeddingBackend,
    allow_hashing: bool = False,
) -> CorpusManifest:
    """Ingest, mask and atomically persist a deduplicated private corpus."""
    output_path = _validated_output_dir(output_dir)
    if embedding_backend.name == "hashing" and not allow_hashing:
        raise ValueError("hashing embeddings require allow_hashing=True")

    instructor = InstructorIdentity(aliases=config.instructor_aliases)
    assistant = Assistant(
        embedding_backend=embedding_backend,
        llm=ExtractiveClient(),
        instructor=instructor,
    )
    if not isinstance(assistant.store, NumpyVectorStore):
        raise ValueError("private corpus persistence requires the numpy vector store")
    input_hashes: list[str] = []
    source_labels: list[str] = []
    seen_hashes: set[str] = set()

    for position, raw_path in enumerate(inputs, start=1):
        input_path = Path(raw_path)
        data = input_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        label = _safe_label(config, input_path, position)
        sources = ingest_bytes(data, input_path.name, instructor=instructor)
        sources = _sanitise_sources(sources, label, config.instructor_aliases)
        assistant.add_sources(sources)
        input_hashes.append(digest)
        source_labels.extend(source.source for source in sources)

    chunks = assistant.store.all_chunks()
    if not any(chunk.is_instructor for chunk in chunks):
        raise ValueError("corpus must contain at least one instructor chunk")
    privacy_scan_clean = _chunks_are_private(assistant, config.instructor_aliases)
    if not privacy_scan_clean:
        raise ValueError("masked corpus failed the privacy scan")

    stats = assistant.stats
    manifest = CorpusManifest(
        schema_version=1,
        embedding_backend=embedding_backend.name,
        embedding_description=embedding_backend.description,
        embedding_dimension=embedding_backend.dimension,
        source_count=stats.file_count,
        message_count=stats.message_count,
        instructor_message_count=stats.instructor_message_count,
        chunk_count=stats.chunk_count,
        masked_entity_count=stats.masked_entity_count,
        source_labels=tuple(source_labels),
        input_hashes=tuple(input_hashes),
        privacy_scan_clean=privacy_scan_clean,
    )
    _write_index_atomically(output_path, assistant.store, manifest)
    return manifest


def load_private_assistant(index_dir: Path, llm: LLMClient | None = None) -> Assistant:
    """Load a persisted index after validating its embedding compatibility."""
    index_path = Path(index_dir)
    manifest = CorpusManifest.load(index_path / "manifest.json")
    backend = resolve_embedding_backend(manifest.embedding_backend)
    if backend.name != manifest.embedding_backend:
        raise ValueError("saved embedding backend does not match the resolved backend")
    if backend.dimension != manifest.embedding_dimension:
        raise ValueError("saved embedding dimension does not match the resolved backend")
    if backend.description != manifest.embedding_description:
        raise ValueError("saved embedding description does not match the resolved backend")

    store = NumpyVectorStore.load(index_path)
    vectors = store.vectors()
    stored_dimension = vectors.shape[1] if vectors is not None and vectors.ndim == 2 else 0
    if stored_dimension != manifest.embedding_dimension:
        raise ValueError("stored vector dimension does not match the corpus manifest")
    if len(store) != manifest.chunk_count:
        raise ValueError("stored chunk count does not match the corpus manifest")

    stats = IndexStats(
        file_count=manifest.source_count,
        message_count=manifest.message_count,
        instructor_message_count=manifest.instructor_message_count,
        chunk_count=manifest.chunk_count,
        masked_entity_count=manifest.masked_entity_count,
        sources=manifest.source_labels,
        embedding_backend=backend.description,
    )
    return Assistant(embedding_backend=backend, llm=llm, store=store, persisted_stats=stats)
