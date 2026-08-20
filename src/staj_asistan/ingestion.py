"""File ingestion.

Accepts what participants actually have on disk: a raw WhatsApp ``.zip`` export,
the ``_chat.txt`` inside it, or plain notes. Media files inside an archive are
never read - only text entries are, which removes the risk of pulling personal
photos into the project.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import ChunkingConfig, DEFAULT_CHUNKING, chunk_document, chunk_messages
from .models import Chunk, Message
from .privacy import DEFAULT_POLICY, MaskingReport, PrivacyPolicy, mask_text
from .whatsapp_parser import (
    DEFAULT_INSTRUCTOR,
    InstructorIdentity,
    looks_like_whatsapp_export,
    parse_whatsapp_export,
)

TEXT_SUFFIXES = {".txt", ".md", ".log"}


@dataclass
class IngestedSource:
    """One logical source (a chat export or a document) after masking."""

    source: str
    kind: str
    messages: list[Message] = field(default_factory=list)
    text: str = ""
    masked_entities: int = 0

    @property
    def message_count(self) -> int:
        return len(self.messages) if self.kind == "whatsapp" else 1

    @property
    def instructor_message_count(self) -> int:
        return sum(1 for m in self.messages if m.is_instructor)

    def to_chunks(self, config: ChunkingConfig = DEFAULT_CHUNKING) -> list[Chunk]:
        if self.kind == "whatsapp":
            return chunk_messages(self.messages, config)
        return chunk_document(self.text, self.source, config)


def _clean_source_name(name: str) -> str:
    stem = Path(name).stem
    stem = stem.replace("WhatsApp Chat - ", "").replace("_chat", "").strip()
    return stem or "kaynak"


def ingest_text(
    text: str,
    source: str,
    policy: PrivacyPolicy = DEFAULT_POLICY,
    instructor: InstructorIdentity = DEFAULT_INSTRUCTOR,
) -> IngestedSource:
    """Route raw text to the WhatsApp parser or to plain-document handling."""
    if looks_like_whatsapp_export(text):
        report = MaskingReport()
        messages = parse_whatsapp_export(
            text, source=source, policy=policy, instructor=instructor, report=report
        )
        return IngestedSource(
            source=source, kind="whatsapp", messages=messages, masked_entities=report.total
        )
    masked = mask_text(text, policy)
    return IngestedSource(
        source=source,
        kind="document",
        text=masked.text,
        masked_entities=masked.report.total,
    )


def ingest_bytes(
    data: bytes,
    filename: str,
    policy: PrivacyPolicy = DEFAULT_POLICY,
    instructor: InstructorIdentity = DEFAULT_INSTRUCTOR,
) -> list[IngestedSource]:
    """Ingest an uploaded file. ``.zip`` archives yield one source per text entry."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        return _ingest_zip(data, filename, policy, instructor)
    text = data.decode("utf-8", errors="replace")
    return [ingest_text(text, _clean_source_name(filename), policy, instructor)]


def _ingest_zip(
    data: bytes, filename: str, policy: PrivacyPolicy, instructor: InstructorIdentity
) -> list[IngestedSource]:
    results: list[IngestedSource] = []
    base = _clean_source_name(filename)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.namelist():
            # Text entries only: images/audio/video in the archive are ignored on purpose.
            if Path(entry).suffix.lower() not in TEXT_SUFFIXES:
                continue
            content = archive.read(entry).decode("utf-8", errors="replace")
            entry_name = base if Path(entry).stem == "_chat" else f"{base}-{Path(entry).stem}"
            results.append(ingest_text(content, entry_name, policy, instructor))
    return results


def ingest_path(
    path: str | Path,
    policy: PrivacyPolicy = DEFAULT_POLICY,
    instructor: InstructorIdentity = DEFAULT_INSTRUCTOR,
) -> list[IngestedSource]:
    """Ingest a single file from disk."""
    file_path = Path(path)
    return ingest_bytes(file_path.read_bytes(), file_path.name, policy, instructor)


def ingest_directory(
    directory: str | Path,
    policy: PrivacyPolicy = DEFAULT_POLICY,
    instructor: InstructorIdentity = DEFAULT_INSTRUCTOR,
) -> list[IngestedSource]:
    """Ingest every supported file in a directory, sorted for reproducibility."""
    root = Path(directory)
    sources: list[IngestedSource] = []
    if not root.exists():
        return sources
    for file_path in sorted(root.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in TEXT_SUFFIXES | {".zip"}:
            sources.extend(ingest_path(file_path, policy, instructor))
    return sources


def sample_directory() -> Path:
    """Location of the anonymised sample data shipped with the repository."""
    return Path(__file__).resolve().parents[2] / "data" / "samples"
