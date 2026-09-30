"""Validation for local-only private corpus configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _private_index_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a path under data/index")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("data", "index"):
        raise ValueError(f"{field_name} must be a path under data/index")
    return path


@dataclass(frozen=True)
class PrivateConfig:
    """Private deployment settings that must never be serialised into an index."""

    instructor_aliases: tuple[str, ...]
    allowed_telegram_chat_ids: frozenset[int] = frozenset()
    source_labels: dict[str, str] = field(default_factory=dict)
    index_dir: Path = Path("data/index/private")
    queue_db: Path = Path("data/index/telegram-queue.sqlite3")

    @classmethod
    def load(cls, path: Path) -> PrivateConfig:
        """Load and validate configuration without echoing sensitive values."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("private configuration could not be read") from exc
        if not isinstance(payload, dict):
            raise ValueError("private configuration must be a JSON object")

        aliases = payload.get("instructor_aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
        ):
            raise ValueError("at least one instructor alias is required")

        chat_ids = payload.get("allowed_telegram_chat_ids", [])
        if not isinstance(chat_ids, list) or any(
            isinstance(chat_id, bool) or not isinstance(chat_id, int) for chat_id in chat_ids
        ):
            raise ValueError("allowed Telegram chat IDs must be integers")

        source_labels = payload.get("source_labels", {})
        if not isinstance(source_labels, dict) or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(label, str)
            or not label.strip()
            for key, label in source_labels.items()
        ):
            raise ValueError("source labels must be non-empty strings")

        return cls(
            instructor_aliases=tuple(alias.strip() for alias in aliases),
            allowed_telegram_chat_ids=frozenset(chat_ids),
            source_labels={key: label.strip() for key, label in source_labels.items()},
            index_dir=_private_index_path(
                payload.get("index_dir", "data/index/private"), "index_dir"
            ),
            queue_db=_private_index_path(
                payload.get("queue_db", "data/index/telegram-queue.sqlite3"), "queue_db"
            ),
        )
