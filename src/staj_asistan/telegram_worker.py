"""Telegram polling worker and the single approval-gated outbound path.

:class:`TelegramPoller` only ever creates queue rows; it holds a transport for
``getUpdates`` and never calls ``send_message``. :class:`ApprovalService` is the
one place a reply leaves the machine, and only after a human-triggered claim.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .approval_queue import ApprovalItem, ApprovalQueue, InvalidTransition
from .privacy import DEFAULT_POLICY, PrivacyPolicy, mask_text, pseudonymize_sender
from .telegram import (
    MAX_MESSAGE_LENGTH,
    TelegramDeliveryRejected,
    TelegramDeliveryUncertain,
    TelegramError,
    TelegramTransport,
)
from .triage import Channel, IncomingMessage, ReplyDecision, TriageResult, triage

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .pipeline import Assistant

logger = logging.getLogger(__name__)

_GROUP_CHAT_TYPES = frozenset({"group", "supergroup"})

TriageFn = Callable[[IncomingMessage, "Assistant"], TriageResult]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class _Accepted:
    chat_id: int
    message_id: int
    sender_id: int
    text: str
    received_at: datetime | None


class TelegramPoller:
    """Turns allowlisted group messages into masked, pending queue rows."""

    def __init__(
        self,
        *,
        queue: ApprovalQueue,
        transport: TelegramTransport,
        assistant: Assistant,
        allowed_chat_ids: Iterable[int],
        policy: PrivacyPolicy = DEFAULT_POLICY,
        triage_fn: TriageFn = triage,
    ) -> None:
        allowed = frozenset(allowed_chat_ids)
        if not allowed or not all(_is_int(chat_id) for chat_id in allowed):
            raise ValueError("at least one allowed integer Telegram chat ID is required")
        self.queue = queue
        self._transport = transport
        self._assistant = assistant
        self._allowed = allowed
        self._policy = policy
        self._triage = triage_fn
        self._known = {item.update_id: item.id for item in queue.list_items()}
        self.offset: int | None = None

    def _advance(self, update_id: int) -> None:
        if self.offset is None or update_id + 1 > self.offset:
            self.offset = update_id + 1

    def _accept(self, update: dict[str, Any]) -> _Accepted | str:
        message = update.get("message")
        if not isinstance(message, dict):
            return "not a new message"
        chat = message.get("chat")
        if not isinstance(chat, dict) or not _is_int(chat.get("id")):
            return "no chat"
        if chat["id"] not in self._allowed:
            return "chat not allowlisted"
        if chat.get("type") not in _GROUP_CHAT_TYPES:
            return "not a group chat"
        sender = message.get("from")
        if not isinstance(sender, dict) or not _is_int(sender.get("id")):
            return "no sender"
        if sender.get("is_bot") is not False or "sender_chat" in message:
            return "bot or channel sender"
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return "not text"
        if not _is_int(message.get("message_id")):
            return "no message id"
        date = message.get("date")
        return _Accepted(
            chat_id=chat["id"],
            message_id=message["message_id"],
            sender_id=sender["id"],
            text=text,
            received_at=datetime.fromtimestamp(date, tz=UTC) if _is_int(date) else None,
        )

    def _group_label(self, chat_id: int) -> str:
        digest = hashlib.sha256(
            f"{self._policy.salt}:telegram-chat:{chat_id}".encode()
        ).hexdigest()[:4]
        return f"Telegram grubu#{digest}"

    def process_update(self, update: dict[str, Any]) -> int | None:
        """Queue one update and return its item ID, or ``None`` when it is ignored.

        The offset advances past every observed update, ignored ones included, but
        only after the update has been fully handled: if triage or the queue write
        raises, the offset stays put so the next poll fetches the update again.
        """

        update_id = update.get("update_id") if isinstance(update, dict) else None
        if not _is_int(update_id):
            logger.debug("Ignored Telegram update without an update_id")
            return None
        if update_id in self._known:
            self._advance(update_id)
            return self._known[update_id]
        if self.offset is not None and update_id < self.offset:
            return None

        accepted = self._accept(update)
        if isinstance(accepted, str):
            logger.debug("Ignored Telegram update %d: %s", update_id, accepted)
            self._advance(update_id)
            return None

        masked = mask_text(accepted.text, self._policy).text
        if not masked.strip():
            logger.debug("Ignored Telegram update %d: empty after masking", update_id)
            self._advance(update_id)
            return None
        group_label = self._group_label(accepted.chat_id)
        sender_alias = pseudonymize_sender(f"telegram:{accepted.sender_id}", self._policy)
        incoming = IncomingMessage(
            text=masked,
            channel=Channel.TELEGRAM,
            group=group_label,
            sender_alias=sender_alias,
            received_at=accepted.received_at,
        )
        result = self._triage(incoming, self._assistant)
        item = self.queue.enqueue(
            update_id=update_id,
            chat_id=accepted.chat_id,
            message_id=accepted.message_id,
            incoming_text=masked,
            sender_alias=sender_alias,
            group_label=group_label,
            triage_result=result,
        )
        self._known[update_id] = item.id
        self._advance(update_id)
        logger.info("Queued Telegram update %d as item %d", update_id, item.id)
        return item.id

    def poll_once(self, timeout: int = 30) -> int:
        """Fetch one batch of updates and return how many were queued."""

        updates = self._transport.get_updates(offset=self.offset, timeout=timeout)
        ordered = sorted(
            updates,
            key=lambda u: u.get("update_id") if _is_int(u.get("update_id")) else -1,
        )
        return sum(1 for update in ordered if self.process_update(update) is not None)

    def run_forever(
        self,
        stop: threading.Event,
        *,
        timeout: int = 30,
        backoff_seconds: float = 5.0,
    ) -> None:
        """Long-poll until ``stop`` is set, backing off after any failure."""

        while not stop.is_set():
            try:
                self.poll_once(timeout=timeout)
            except TelegramError as exc:
                logger.warning("Telegram polling failed: %s", exc)
                stop.wait(backoff_seconds)
            except Exception as exc:  # noqa: BLE001 - worker must survive and stay quiet
                logger.warning("Telegram update handling failed (%s)", type(exc).__name__)
                stop.wait(backoff_seconds)


class ApprovalService:
    """The only outbound path: claim, send the reviewed text once, record the outcome."""

    def __init__(self, *, queue: ApprovalQueue, transport: TelegramTransport) -> None:
        self.queue = queue
        self._transport = transport

    def approve_and_send(self, item_id: int, edited_draft: str) -> ApprovalItem:
        """Send a human-reviewed draft for one queue item, at most once.

        Returns the ``sent`` item, or the ``failed`` item when Telegram definitely
        did not deliver (another explicit approval may retry it). When delivery is
        uncertain the item stays ``sending``, which no claim can reopen, and
        :class:`TelegramDeliveryUncertain` is raised for a human to reconcile.
        """

        if len(edited_draft) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Reviewed draft exceeds {MAX_MESSAGE_LENGTH} characters")

        claimed = self.queue.claim_for_send(item_id, edited_draft)
        if claimed.reply_decision is ReplyDecision.DO_NOT_ANSWER:
            self.queue.mark_failed(item_id, "DO_NOT_ANSWER items cannot be sent")
            raise InvalidTransition(f"Item {item_id} cannot be sent")

        try:
            self._transport.send_message(claimed.chat_id, claimed.draft, claimed.message_id)
        except TelegramDeliveryRejected as exc:
            logger.warning("Telegram delivery rejected for item %d", item_id)
            return self.queue.mark_failed(item_id, str(exc))
        except TelegramDeliveryUncertain:
            logger.warning("Telegram delivery uncertain for item %d", item_id)
            raise TelegramDeliveryUncertain(
                f"Item {item_id} may have been delivered; left in sending for manual review"
            ) from None
        except Exception as exc:  # noqa: BLE001 - unknown outcome must never become retryable
            logger.warning(
                "Telegram delivery raised %s for item %d", type(exc).__name__, item_id
            )
            raise TelegramDeliveryUncertain(
                f"Item {item_id} may have been delivered; left in sending for manual review"
            ) from None
        return self.queue.mark_sent(item_id)
