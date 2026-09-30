"""Durable SQLite queue for replies that require an explicit human approval."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from staj_asistan.models import (
    AssistantMode,
    AuthorRole,
    ChunkCategory,
    Citation,
    EvidenceLevel,
)
from staj_asistan.triage import MessageIntent, ReplyDecision, TriageResult


class QueueStatus(StrEnum):
    """Lifecycle states for one queued reply."""

    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    REJECTED = "rejected"


class InvalidTransition(RuntimeError):
    """Raised when an item cannot move from its current state as requested."""


@dataclass(frozen=True)
class ApprovalItem:
    """Privacy-safe queue record shown to an approver."""

    id: int
    update_id: int
    chat_id: int
    message_id: int
    incoming_text: str
    sender_alias: str
    group_label: str
    intent: MessageIntent
    selected_mode: AssistantMode
    evidence_level: EvidenceLevel
    source_categories: tuple[ChunkCategory, ...]
    reply_decision: ReplyDecision
    reason: str
    citations: tuple[Citation, ...]
    draft: str
    status: QueueStatus
    error: str
    created_at: datetime
    updated_at: datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    update_id INTEGER NOT NULL UNIQUE,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    incoming_text TEXT NOT NULL,
    sender_alias TEXT NOT NULL,
    group_label TEXT NOT NULL,
    intent TEXT NOT NULL,
    selected_mode TEXT NOT NULL,
    evidence_level TEXT NOT NULL,
    source_categories TEXT NOT NULL,
    reply_decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    citations TEXT NOT NULL,
    draft TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_BOT_TOKEN = re.compile(r"\b(?:bot)?\d{6,}:[A-Za-z0-9_-]{20,}\b", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sanitize_error(error: str) -> str:
    cleaned = _URL.sub("[redacted-url]", str(error))
    cleaned = _BOT_TOKEN.sub("[redacted-token]", cleaned)
    cleaned = _CONTROL.sub(" ", cleaned)
    return " ".join(cleaned.split())[:240]


def _citation_to_dict(citation: Citation) -> dict[str, object]:
    return {
        "index": citation.index,
        "label": citation.label,
        "source": citation.source,
        "role": citation.role.value,
        "snippet": citation.snippet,
        "score": citation.score,
    }


def _citation_from_dict(value: dict[str, object]) -> Citation:
    return Citation(
        index=int(value["index"]),
        label=str(value["label"]),
        source=str(value["source"]),
        role=AuthorRole(str(value["role"])),
        snippet=str(value["snippet"]),
        score=float(value["score"]),
    )


class ApprovalQueue:
    """A small SQLite repository with guarded queue-state transitions."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def enqueue(
        self,
        *,
        update_id: int,
        chat_id: int,
        message_id: int,
        incoming_text: str,
        sender_alias: str,
        group_label: str,
        triage_result: TriageResult,
    ) -> ApprovalItem:
        """Insert once per Telegram update and return the durable record."""

        timestamp = _now()
        categories = json.dumps(
            [category.value for category in triage_result.source_categories],
            ensure_ascii=False,
        )
        citations = json.dumps(
            [_citation_to_dict(citation) for citation in triage_result.answer.citations],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        values = (
            update_id,
            chat_id,
            message_id,
            incoming_text,
            sender_alias,
            group_label,
            triage_result.intent.value,
            triage_result.selected_mode.value,
            triage_result.evidence_level.value,
            categories,
            triage_result.reply_decision.value,
            triage_result.reason,
            citations,
            triage_result.draft_reply,
            QueueStatus.PENDING.value,
            timestamp,
            timestamp,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO approval_items (
                    update_id, chat_id, message_id, incoming_text, sender_alias,
                    group_label, intent, selected_mode, evidence_level,
                    source_categories, reply_decision, reason, citations, draft,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(update_id) DO NOTHING
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM approval_items WHERE update_id = ?",
                (update_id,),
            ).fetchone()
        assert row is not None
        return self._to_item(row)

    def list_items(self, status: QueueStatus | None = None) -> list[ApprovalItem]:
        """List records oldest first, optionally restricted to one status."""

        with self._connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT * FROM approval_items ORDER BY id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM approval_items WHERE status = ? ORDER BY id",
                    (status.value,),
                ).fetchall()
        return [self._to_item(row) for row in rows]

    def get(self, item_id: int) -> ApprovalItem:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM approval_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            raise KeyError(item_id)
        return self._to_item(row)

    def reject(self, item_id: int) -> ApprovalItem:
        """Reject a pending item permanently."""

        return self._transition(
            item_id,
            from_status=QueueStatus.PENDING,
            to_status=QueueStatus.REJECTED,
        )

    def claim_for_send(self, item_id: int, edited_draft: str) -> ApprovalItem:
        """Atomically reserve an answer for one outbound send attempt."""

        if not edited_draft.strip():
            raise ValueError("Reviewed draft must not be empty")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            timestamp = _now()
            cursor = connection.execute(
                """
                UPDATE approval_items
                SET status = ?, draft = ?, error = '', updated_at = ?
                WHERE id = ?
                  AND status IN (?, ?)
                  AND reply_decision != ?
                """,
                (
                    QueueStatus.SENDING.value,
                    edited_draft,
                    timestamp,
                    item_id,
                    QueueStatus.PENDING.value,
                    QueueStatus.FAILED.value,
                    ReplyDecision.DO_NOT_ANSWER.value,
                ),
            )
            if cursor.rowcount != 1:
                self._raise_claim_error(connection, item_id)
            row = connection.execute(
                "SELECT * FROM approval_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        assert row is not None
        return self._to_item(row)

    def mark_sent(self, item_id: int) -> ApprovalItem:
        """Complete a claimed item after its outbound call succeeds."""

        return self._transition(
            item_id,
            from_status=QueueStatus.SENDING,
            to_status=QueueStatus.SENT,
        )

    def mark_failed(self, item_id: int, error: str) -> ApprovalItem:
        """Return a claimed item to a retryable state with a safe error."""

        return self._transition(
            item_id,
            from_status=QueueStatus.SENDING,
            to_status=QueueStatus.FAILED,
            error=_sanitize_error(error),
        )

    def _transition(
        self,
        item_id: int,
        *,
        from_status: QueueStatus,
        to_status: QueueStatus,
        error: str = "",
    ) -> ApprovalItem:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE approval_items
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (to_status.value, error, _now(), item_id, from_status.value),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM approval_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
                if exists is None:
                    raise KeyError(item_id)
                raise InvalidTransition(
                    f"Item {item_id} cannot transition from {from_status.value}"
                )
            row = connection.execute(
                "SELECT * FROM approval_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        assert row is not None
        return self._to_item(row)

    @staticmethod
    def _raise_claim_error(connection: sqlite3.Connection, item_id: int) -> None:
        row = connection.execute(
            "SELECT status, reply_decision FROM approval_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            raise KeyError(item_id)
        raise InvalidTransition(
            f"Item {item_id} cannot be claimed from {row['status']}"
        )

    @staticmethod
    def _to_item(row: sqlite3.Row) -> ApprovalItem:
        category_values = json.loads(row["source_categories"])
        citation_values = json.loads(row["citations"])
        return ApprovalItem(
            id=row["id"],
            update_id=row["update_id"],
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            incoming_text=row["incoming_text"],
            sender_alias=row["sender_alias"],
            group_label=row["group_label"],
            intent=MessageIntent(row["intent"]),
            selected_mode=AssistantMode(row["selected_mode"]),
            evidence_level=EvidenceLevel(row["evidence_level"]),
            source_categories=tuple(ChunkCategory(value) for value in category_values),
            reply_decision=ReplyDecision(row["reply_decision"]),
            reason=row["reason"],
            citations=tuple(_citation_from_dict(value) for value in citation_values),
            draft=row["draft"],
            status=QueueStatus(row["status"]),
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
