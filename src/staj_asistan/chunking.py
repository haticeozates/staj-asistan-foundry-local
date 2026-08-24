"""Turns messages into retrievable chunks.

Two shapes of content need different treatment:

1. **Long instructor announcements** (up to 19k characters in the real export)
   are split into overlapping, paragraph-aware windows.
2. **Short conversational messages** are grouped into time-bounded windows.
   This matters more than it sounds: many instructor answers are two words
   ("Sertifika tek", "Sadece email, link") and are meaningless without the
   participant question they reply to. Keeping the exchange together is what
   makes those answers retrievable and quotable.

Every chunk keeps the metadata needed to cite it and to weight it later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

from .models import AuthorRole, Chunk, Message
from .topics import classify_chunk


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 900
    overlap_chars: int = 150
    conversation_gap_minutes: int = 30
    max_messages_per_chunk: int = 12
    min_chunk_chars: int = 8

    def __post_init__(self) -> None:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")


DEFAULT_CHUNKING = ChunkingConfig()

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+")


def format_message(message: Message) -> str:
    """Render a message the way it will appear inside a chunk and a citation."""
    when = message.timestamp.strftime("%d.%m.%Y %H:%M") if message.timestamp else "tarihsiz"
    return f"[{when}] {message.sender}: {message.text}"


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split on paragraph, then sentence, then hard boundaries, keeping an overlap."""
    if len(text) <= max_chars:
        return [text]

    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            while len(sentence) > max_chars:
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars - overlap_chars :]
            if sentence:
                units.append(sentence)

    windows: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        windows.append(current)
        tail = current[-overlap_chars:] if overlap_chars else ""
        current = f"{tail}\n\n{unit}".strip() if tail else unit
    if current:
        windows.append(current)
    return windows


def _dominant_role(messages: list[Message]) -> tuple[AuthorRole, float]:
    if not messages:
        return AuthorRole.PARTICIPANT, 0.0
    instructor_chars = sum(len(m.text) for m in messages if m.is_instructor)
    total_chars = sum(len(m.text) for m in messages) or 1
    ratio = instructor_chars / total_chars
    return (AuthorRole.INSTRUCTOR if ratio >= 0.5 else AuthorRole.PARTICIPANT), ratio


def _make_chunk(
    chunk_id: str,
    text: str,
    messages: list[Message],
    source: str,
    role: AuthorRole | None = None,
    instructor_ratio: float | None = None,
) -> Chunk:
    computed_role, computed_ratio = _dominant_role(messages)
    timestamps = [m.timestamp for m in messages if m.timestamp]
    senders: list[str] = []
    for message in messages:
        if message.sender and message.sender not in senders:
            senders.append(message.sender)
    body = text.strip()
    return Chunk(
        chunk_id=chunk_id,
        text=body,
        source=source,
        role=role or computed_role,
        senders=tuple(senders),
        start_time=min(timestamps) if timestamps else None,
        end_time=max(timestamps) if timestamps else None,
        message_count=len(messages),
        instructor_ratio=computed_ratio if instructor_ratio is None else instructor_ratio,
        category=classify_chunk(body, source),
    )


def _starts_new_window(current: list[Message], message: Message, config: ChunkingConfig) -> bool:
    if not current:
        return False
    previous = current[-1]
    if previous.source != message.source:
        return True
    if len(current) >= config.max_messages_per_chunk:
        return True
    if previous.timestamp and message.timestamp:
        gap = message.timestamp - previous.timestamp
        if gap > timedelta(minutes=config.conversation_gap_minutes) or gap < timedelta(0):
            return True
    current_chars = sum(len(format_message(m)) for m in current)
    return current_chars + len(format_message(message)) > config.max_chars


def chunk_messages(
    messages: list[Message], config: ChunkingConfig = DEFAULT_CHUNKING
) -> list[Chunk]:
    """Group and split messages into chunks, preserving citation metadata."""
    chunks: list[Chunk] = []
    window: list[Message] = []
    counters: dict[str, int] = {}

    def next_id(source: str) -> str:
        counters[source] = counters.get(source, 0) + 1
        return f"{source}#{counters[source]:04d}"

    def flush() -> None:
        nonlocal window
        if not window:
            return
        text = "\n".join(format_message(m) for m in window)
        if len(text) >= config.min_chunk_chars:
            source = window[0].source
            chunks.append(_make_chunk(next_id(source), text, list(window), source))
        window = []

    for message in messages:
        if message.is_system_event:
            continue
        rendered = format_message(message)
        if len(rendered) > config.max_chars:
            flush()
            header = (
                f"[{message.timestamp.strftime('%d.%m.%Y %H:%M')}] {message.sender}:"
                if message.timestamp
                else f"{message.sender}:"
            )
            for part in _split_long_text(message.text, config.max_chars, config.overlap_chars):
                chunks.append(
                    _make_chunk(
                        next_id(message.source),
                        f"{header}\n{part}",
                        [message],
                        message.source,
                    )
                )
            continue
        if _starts_new_window(window, message, config):
            flush()
        window.append(message)
    flush()
    return chunks


def chunk_document(
    text: str, source: str, config: ChunkingConfig = DEFAULT_CHUNKING
) -> list[Chunk]:
    """Chunk a plain document (notes, e-mail export) that has no message structure."""
    chunks: list[Chunk] = []
    for index, part in enumerate(
        _split_long_text(text.strip(), config.max_chars, config.overlap_chars), start=1
    ):
        if len(part.strip()) < config.min_chunk_chars:
            continue
        body = part.strip()
        chunks.append(
            Chunk(
                chunk_id=f"{source}#{index:04d}",
                text=body,
                source=source,
                role=AuthorRole.DOCUMENT,
                message_count=1,
                category=classify_chunk(body, source),
            )
        )
    return chunks
