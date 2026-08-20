"""Core domain objects shared by every layer of the pipeline.

Everything that travels between ingestion, retrieval and generation is defined
here so that the layers stay decoupled and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AuthorRole(str, Enum):
    """Who produced a message. Drives authority weighting during retrieval.

    ``DOCUMENT`` covers curated notes and summaries: not the instructor's own words,
    but not an off-hand guess in the group either, so retrieval treats it as neutral.
    """

    INSTRUCTOR = "instructor"
    PARTICIPANT = "participant"
    DOCUMENT = "document"
    SYSTEM = "system"


class AssistantMode(str, Enum):
    """The four operating modes exposed by the assistant."""

    INSTRUCTOR_QA = "instructor_qa"
    SUBMISSION_CHECKLIST = "submission_checklist"
    CORRECTION_ANALYZER = "correction_analyzer"
    TECHNICAL_HELP = "technical_help"


class EvidenceLevel(str, Enum):
    """How much the produced answer can be trusted, given the retrieved sources."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass(frozen=True)
class Message:
    """A single chat/document message after parsing and privacy masking."""

    text: str
    sender: str
    role: AuthorRole
    source: str
    timestamp: datetime | None = None
    line_no: int = 0
    is_system_event: bool = False

    @property
    def is_instructor(self) -> bool:
        return self.role is AuthorRole.INSTRUCTOR

    @property
    def date_label(self) -> str:
        return self.timestamp.strftime("%d.%m.%Y") if self.timestamp else "tarihsiz"


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text plus the metadata needed to cite it."""

    chunk_id: str
    text: str
    source: str
    role: AuthorRole
    senders: tuple[str, ...] = ()
    start_time: datetime | None = None
    end_time: datetime | None = None
    message_count: int = 1
    instructor_ratio: float = 0.0

    @property
    def is_instructor(self) -> bool:
        return self.role is AuthorRole.INSTRUCTOR

    @property
    def label(self) -> str:
        """Short human-readable citation label, e.g. ``grup1 · Eğitmen · 24.07.2026``."""
        who = {
            AuthorRole.INSTRUCTOR: "Eğitmen",
            AuthorRole.DOCUMENT: "Belge",
            AuthorRole.SYSTEM: "Sistem",
        }.get(self.role, "Katılımcı")
        when = self.start_time.strftime("%d.%m.%Y") if self.start_time else "tarihsiz"
        return f"{self.source} · {who} · {when}"


@dataclass
class ScoredChunk:
    """A chunk together with the reason it was retrieved."""

    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    authority_boost: float = 0.0
    recency_boost: float = 0.0

    @property
    def explanation(self) -> str:
        parts = [f"anlamsal={self.dense_score:.2f}", f"kelime={self.lexical_score:.2f}"]
        if self.authority_boost > 0:
            parts.append(f"eğitmen+{self.authority_boost:.2f}")
        elif self.authority_boost < 0:
            parts.append(f"katılımcı{self.authority_boost:.2f}")
        if self.recency_boost:
            parts.append(f"güncellik+{self.recency_boost:.2f}")
        return ", ".join(parts)


@dataclass
class Citation:
    """A source shown underneath an answer."""

    index: int
    label: str
    source: str
    role: AuthorRole
    snippet: str
    score: float

    @property
    def marker(self) -> str:
        return f"[{self.index}]"


@dataclass
class Answer:
    """Final assistant output handed to the UI."""

    text: str
    mode: AssistantMode
    evidence: EvidenceLevel
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    structured: dict | None = None
    generator: str = "unknown"
    grounded: bool = True

    @property
    def has_sources(self) -> bool:
        return bool(self.citations)


@dataclass
class IndexStats:
    """Sidebar statistics describing what is currently indexed."""

    file_count: int = 0
    message_count: int = 0
    instructor_message_count: int = 0
    chunk_count: int = 0
    masked_entity_count: int = 0
    sources: tuple[str, ...] = ()
    embedding_backend: str = "-"

    def as_dict(self) -> dict[str, object]:
        return {
            "Dosya": self.file_count,
            "Mesaj": self.message_count,
            "Eğitmen mesajı": self.instructor_message_count,
            "Chunk": self.chunk_count,
            "Maskelenen veri": self.masked_entity_count,
        }
