"""Telegram Bot API boundary.

This is the only module that knows how to talk to Telegram. Two rules shape it:

* The bot token is read from ``TELEGRAM_BOT_TOKEN`` and never leaves this object.
  Request URLs embed the token, so no exception, log record or message built here
  may contain a URL or a raw ``requests`` error string.
* A failed ``sendMessage`` is classified conservatively. Only failures that prove
  Telegram never accepted the request raise :class:`TelegramDeliveryRejected`;
  anything that might have been delivered raises :class:`TelegramDeliveryUncertain`
  so that nobody re-sends a message the group may already have seen.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import Any, Protocol

import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
API_ROOT = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 4096

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_BOT_TOKEN = re.compile(r"(?:bot)?\d{6,}:[A-Za-z0-9_-]{20,}", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


class TelegramError(RuntimeError):
    """A Telegram failure whose message is safe to log, store and display."""


class TelegramDeliveryRejected(TelegramError):
    """``sendMessage`` definitely did not deliver; an explicit retry is safe."""


class TelegramDeliveryUncertain(TelegramError):
    """``sendMessage`` may have delivered; retrying could duplicate the reply."""


class TelegramTransport(Protocol):
    """What the poller and the approval service need from Telegram."""

    def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]: ...

    def send_message(self, chat_id: int, text: str, reply_to_message_id: int) -> int: ...


def _safe(text: object, token: str = "") -> str:
    cleaned = str(text)
    if token:
        cleaned = cleaned.replace(token, "[redacted-token]")
    cleaned = _URL.sub("[redacted-url]", cleaned)
    cleaned = _BOT_TOKEN.sub("[redacted-token]", cleaned)
    cleaned = _CONTROL.sub(" ", cleaned)
    return " ".join(cleaned.split())[:200]


class _TokenRedactingFilter(logging.Filter):
    """Scrubs tokens from HTTP client debug logs, which print request paths."""

    def __init__(self) -> None:
        super().__init__()
        self.tokens: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if any(token in message for token in self.tokens):
            for token in self.tokens:
                message = message.replace(token, "[redacted-token]")
            record.msg, record.args = message, ()
        return True


_LOG_FILTER = _TokenRedactingFilter()
_HTTP_LOGGERS = ("urllib3.connectionpool", "urllib3.util.retry", "requests")


def _register_token_for_redaction(token: str) -> None:
    _LOG_FILTER.tokens.add(token)
    for name in _HTTP_LOGGERS:
        logger = logging.getLogger(name)
        if _LOG_FILTER not in logger.filters:
            logger.addFilter(_LOG_FILTER)


def _connection_never_opened(exc: requests.ConnectionError) -> bool:
    if isinstance(exc, requests.ConnectTimeout):
        return True
    reason = exc.args[0] if exc.args else None
    return isinstance(reason, MaxRetryError) and isinstance(reason.reason, NewConnectionError)


class TelegramHTTPTransport:
    """Bot API client over ``requests`` with bounded timeouts and sanitised errors."""

    def __init__(
        self,
        token: str,
        *,
        session: Any | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
    ) -> None:
        token = (token or "").strip()
        if not token:
            raise TelegramError(f"{TOKEN_ENV} is not set")
        self._token = token
        self._session = session if session is not None else requests.Session()
        self._connect_timeout = float(connect_timeout)
        self._read_timeout = float(read_timeout)
        _register_token_for_redaction(token)

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, **kwargs: Any
    ) -> TelegramHTTPTransport:
        """Build a transport from ``TELEGRAM_BOT_TOKEN``; the only supported token source."""

        env = os.environ if environ is None else environ
        return cls(env.get(TOKEN_ENV, ""), **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(token=[redacted])"

    __str__ = __repr__

    def close(self) -> None:
        self._session.close()

    def _post(self, method: str, payload: dict[str, Any], read_timeout: float) -> Any:
        return self._session.post(
            f"{API_ROOT}/bot{self._token}/{method}",
            json=payload,
            timeout=(self._connect_timeout, read_timeout),
        )

    def _api_error(self, method: str, response: Any) -> str:
        try:
            body = response.json()
        except ValueError:
            body = None
        description = body.get("description", "") if isinstance(body, dict) else ""
        detail = _safe(description, self._token)
        return f"Telegram {method} failed (HTTP {response.status_code}) {detail}".strip()

    def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": int(timeout), "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = int(offset)
        try:
            response = self._post("getUpdates", payload, int(timeout) + self._read_timeout)
        except requests.RequestException as exc:
            raise TelegramError(f"Telegram getUpdates failed ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise TelegramError(self._api_error("getUpdates", response)) from None
        try:
            body = response.json()
        except ValueError:
            raise TelegramError("Telegram getUpdates returned an unreadable body") from None
        if not isinstance(body, dict) or body.get("ok") is not True:
            raise TelegramError("Telegram getUpdates returned an error body")
        result = body.get("result")
        if not isinstance(result, list):
            raise TelegramError("Telegram getUpdates returned no update list")
        return [update for update in result if isinstance(update, dict)]

    def send_message(self, chat_id: int, text: str, reply_to_message_id: int) -> int:
        payload = {
            "chat_id": int(chat_id),
            "text": text,
            "reply_parameters": {"message_id": int(reply_to_message_id)},
        }
        try:
            response = self._post("sendMessage", payload, self._read_timeout)
        except requests.ConnectionError as exc:
            kind = type(exc).__name__
            if _connection_never_opened(exc):
                raise TelegramDeliveryRejected(
                    f"Telegram sendMessage not delivered ({kind})"
                ) from None
            raise TelegramDeliveryUncertain(
                f"Telegram sendMessage outcome unknown ({kind})"
            ) from None
        except requests.RequestException as exc:
            raise TelegramDeliveryUncertain(
                f"Telegram sendMessage outcome unknown ({type(exc).__name__})"
            ) from None

        status = response.status_code
        if 400 <= status < 500:
            raise TelegramDeliveryRejected(self._api_error("sendMessage", response)) from None
        if status != 200:
            raise TelegramDeliveryUncertain(self._api_error("sendMessage", response)) from None
        try:
            body = response.json()
        except ValueError:
            raise TelegramDeliveryUncertain(
                "Telegram sendMessage outcome unknown (unreadable body)"
            ) from None
        ok = isinstance(body, dict) and body.get("ok") is True
        result = body.get("result") if ok else None
        message_id = result.get("message_id") if isinstance(result, dict) else None
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            raise TelegramDeliveryUncertain(
                "Telegram sendMessage outcome unknown (unexpected body)"
            )
        return message_id
