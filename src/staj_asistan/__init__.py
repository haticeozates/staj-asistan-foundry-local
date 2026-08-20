"""StajAsistan 2.0 - AI Innovators Knowledge & Submission Assistant.

A privacy-aware, citation-grounded local RAG assistant built on Microsoft
Foundry Local. See ``README.md`` for the architecture overview.
"""

from __future__ import annotations

__version__ = "2.0.0"

from .models import (
    Answer,
    AssistantMode,
    AuthorRole,
    Chunk,
    Citation,
    EvidenceLevel,
    IndexStats,
    Message,
    ScoredChunk,
)

__all__ = [
    "Answer",
    "AssistantMode",
    "AuthorRole",
    "Chunk",
    "Citation",
    "EvidenceLevel",
    "IndexStats",
    "Message",
    "ScoredChunk",
    "__version__",
]
