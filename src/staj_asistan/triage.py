"""Triage for a message that just arrived in a programme group.

The rest of the package answers a question that somebody deliberately typed into
the assistant. This module models the other direction, which is the actual
operational problem: a message lands in a WhatsApp or Telegram group, and
somebody has to decide whether it can be answered from the archive at all, who
is allowed to answer it, and what that answer should say.

Three decisions are possible and none of them is "send":

``DRAFT_REPLY``
    The archive answers this, with citations. A human copies the draft.
``NEEDS_HUMAN_APPROVAL``
    There is a draft, but the topic or the evidence means a person must read it
    first — corrections, certificates, deadlines, anything carrying personal data.
``DO_NOT_ANSWER``
    The archive does not support an answer. No draft is produced at all.

Outbound delivery is not implemented here and is not meant to be. The channel on
an :class:`IncomingMessage` is metadata used for display and audit; nothing in
this module reads credentials, opens a socket or talks to a messaging provider.

Intent detection deliberately reuses the heuristics that already exist in
``workflows`` and ``topics`` rather than growing a second, competing set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from .models import Answer, AssistantMode, ChunkCategory, EvidenceLevel
from .topics import infer_query_topic
from .workflows import (
    is_submission_channel_query,
    is_submission_requirement_query,
    suggest_mode,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .pipeline import Assistant


class Channel(str, Enum):
    """Where a message came from. Display and audit metadata only."""

    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    MANUAL = "manual"


class MessageIntent(str, Enum):
    """What the incoming message is operationally asking for."""

    SUBMISSION_REQUIREMENT = "submission_requirement"
    SUBMISSION_CHANNEL = "submission_channel"
    ROSTER_CONTINUE = "roster_continue"
    CORRECTION_REQUEST = "correction_request"
    TECHNICAL_HELP = "technical_help"
    CERTIFICATE = "certificate"
    UNKNOWN = "unknown"


class ReplyDecision(str, Enum):
    """What may happen to the draft. There is deliberately no "send" member."""

    DRAFT_REPLY = "draft_reply"
    NEEDS_HUMAN_APPROVAL = "needs_human_approval"
    DO_NOT_ANSWER = "do_not_answer"


@dataclass(frozen=True)
class IncomingMessage:
    """A message observed in a group, already pseudonymised by the caller.

    ``sender_alias`` is a pseudonym such as ``Katılımcı#a3f2``. A real display
    name must never be put here: the whole pipeline is built so that personal
    data does not travel past ingestion.
    """

    text: str
    channel: Channel = Channel.MANUAL
    group: str = ""
    sender_alias: str = ""
    received_at: datetime | None = None


@dataclass
class TriageResult:
    """The triage verdict handed to the UI."""

    intent: MessageIntent
    selected_mode: AssistantMode
    answer: Answer
    evidence_level: EvidenceLevel
    source_categories: tuple[ChunkCategory, ...]
    reply_decision: ReplyDecision
    reason: str
    draft_reply: str = ""
    message: IncomingMessage | None = None

    @property
    def requires_human(self) -> bool:
        """True whenever a person must act before anything reaches the group.

        Every branch qualifies: even a clean draft has to be copied out by hand.
        """
        return True

    def as_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "selected_mode": self.selected_mode.value,
            "evidence_level": self.evidence_level.value,
            "source_categories": [c.value for c in self.source_categories],
            "reply_decision": self.reply_decision.value,
            "reason": self.reason,
            "channel": self.message.channel.value if self.message else None,
            # The assistant drafts; a human sends. This is a product decision,
            # not a missing feature, so it is reported on every single result.
            "auto_send": False,
        }


# --------------------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------------------

MODE_FOR_INTENT: dict[MessageIntent, AssistantMode] = {
    MessageIntent.SUBMISSION_REQUIREMENT: AssistantMode.SUBMISSION_CHECKLIST,
    MessageIntent.SUBMISSION_CHANNEL: AssistantMode.SUBMISSION_CHECKLIST,
    MessageIntent.CORRECTION_REQUEST: AssistantMode.CORRECTION_ANALYZER,
    MessageIntent.TECHNICAL_HELP: AssistantMode.TECHNICAL_HELP,
    MessageIntent.ROSTER_CONTINUE: AssistantMode.INSTRUCTOR_QA,
    MessageIntent.CERTIFICATE: AssistantMode.INSTRUCTOR_QA,
    MessageIntent.UNKNOWN: AssistantMode.INSTRUCTOR_QA,
}

_CERTIFICATE_HINTS = ("sertifika", "certificate")


def _fold(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def classify_intent(text: str) -> MessageIntent:
    """Label an incoming message by wrapping the existing routing heuristics.

    Order matters. The narrow, high-precision checks (channel, submission rules,
    correction) run before the broad topic inference, because the roster
    vocabulary overlaps almost everything: "listede projem yanlış" is a
    correction, not a roster question, even though it contains "liste".
    """
    if not text.strip():
        return MessageIntent.UNKNOWN

    if is_submission_channel_query(text):
        return MessageIntent.SUBMISSION_CHANNEL
    if is_submission_requirement_query(text):
        return MessageIntent.SUBMISSION_REQUIREMENT

    hinted = suggest_mode(text)
    if hinted is AssistantMode.CORRECTION_ANALYZER:
        return MessageIntent.CORRECTION_REQUEST
    if any(hint in _fold(text) for hint in _CERTIFICATE_HINTS):
        return MessageIntent.CERTIFICATE
    if hinted is AssistantMode.TECHNICAL_HELP:
        return MessageIntent.TECHNICAL_HELP
    if hinted is AssistantMode.SUBMISSION_CHECKLIST:
        return MessageIntent.SUBMISSION_REQUIREMENT

    # Falls back to the retrieval-side topic vocabulary, which covers phrasings the
    # mode hints miss — "çalışmazsa" is technical there but not in the hint list.
    topic = infer_query_topic(text, AssistantMode.INSTRUCTOR_QA)
    if topic is ChunkCategory.CORRECTION:
        return MessageIntent.CORRECTION_REQUEST
    if topic is ChunkCategory.TECHNICAL:
        return MessageIntent.TECHNICAL_HELP
    if topic is ChunkCategory.ROSTER:
        return MessageIntent.ROSTER_CONTINUE
    if topic is ChunkCategory.SUBMISSION:
        return MessageIntent.SUBMISSION_REQUIREMENT
    return MessageIntent.UNKNOWN


# --------------------------------------------------------------------------------------
# Reply policy
# --------------------------------------------------------------------------------------

#: Topics where a wrong answer costs a participant something they cannot undo:
#: a certificate printed with the wrong name, a missed deadline, a stamped form.
#: Erring towards human review here is the point of the mode.
_SENSITIVE_TOPIC_HINTS: tuple[tuple[str, str], ...] = (
    ("sertifika", "sertifika"),
    ("certificate", "sertifika"),
    ("son tarih", "tarih/son teslim"),
    ("deadline", "tarih/son teslim"),
    ("uzatma", "tarih/son teslim"),
    ("uzatabilir", "tarih/son teslim"),
    ("tarih değiş", "tarih/son teslim"),
    ("tarihimi", "tarih/son teslim"),
    ("belge", "resmi belge"),
    ("evrak", "resmi belge"),
    ("imza", "resmi belge"),
    ("kaşe", "resmi belge"),
    ("transkript", "resmi belge"),
    ("staj formu", "resmi belge"),
    ("kaydım", "kişisel kayıt"),
)

#: Masking placeholders plus the raw shapes the masker would have replaced. An
#: incoming message is a query, not indexed content, so it is never masked —
#: if somebody pastes their address into the group, a human handles the reply.
_PERSONAL_DATA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[E-POSTA\]|\[TELEFON\]|\[ÖZEL LİNK\]|\[KULLANICI\]|\[EĞİTMEN E-POSTA\]"),
    re.compile(r"\[KATILIMCI LİSTESİ GİZLENDİ\]"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"(?:\+90|0)\s*\d{3}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}"),
    re.compile(r"\b\d{11}\b"),
)

#: Answers assembled from the canonical rule cards rather than from retrieval.
_RULE_BACKED_KINDS = frozenset({"teslim_kurallari", "teslim_kanali"})

_REDIRECT = "Kaynaklarda karşılık yok; soruyu eğitmene yönlendir."


def _sensitive_topic(text: str) -> str | None:
    lowered = _fold(text)
    for hint, label in _SENSITIVE_TOPIC_HINTS:
        if hint in lowered:
            return label
    return None


def _carries_personal_data(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PERSONAL_DATA_PATTERNS)


def _is_rule_backed(answer: Answer) -> bool:
    return bool(answer.structured) and answer.structured.get("tur") in _RULE_BACKED_KINDS


def decide(
    message: IncomingMessage, intent: MessageIntent, answer: Answer
) -> tuple[ReplyDecision, str]:
    """Apply the reply policy. Returns the decision and the reason to show a human.

    Precedence is explicit because the rules genuinely overlap: a correction
    request with strong evidence still needs a person, and a message with no
    evidence at all has nothing to escalate.
    """
    if answer.evidence is EvidenceLevel.NONE:
        return ReplyDecision.DO_NOT_ANSWER, _REDIRECT

    if intent is MessageIntent.CORRECTION_REQUEST:
        return (
            ReplyDecision.NEEDS_HUMAN_APPROVAL,
            "Düzeltme isteği: listeyi yalnızca eğitmen güncelleyebilir, taslak onay bekler.",
        )

    sensitive = _sensitive_topic(message.text)
    if sensitive:
        return (
            ReplyDecision.NEEDS_HUMAN_APPROVAL,
            f"Hassas konu ({sensitive}): yanlış cevabın geri alınması zor, insan onayı gerekir.",
        )

    if _carries_personal_data(message.text):
        return (
            ReplyDecision.NEEDS_HUMAN_APPROVAL,
            "Mesaj kişisel veri içeriyor; yanıtı bir insan kontrol etmeli.",
        )

    if answer.evidence is EvidenceLevel.LOW:
        if _is_rule_backed(answer):
            return (
                ReplyDecision.NEEDS_HUMAN_APPROVAL,
                "Cevap kural kartından geliyor ama kaynak eşleşmesi zayıf; "
                "insan doğrulaması gerekir.",
            )
        return ReplyDecision.DO_NOT_ANSWER, "Kanıt zayıf; " + _REDIRECT

    return (
        ReplyDecision.DRAFT_REPLY,
        "Kaynaklar soruyu karşılıyor; taslak kopyalanmaya hazır, göndermeyi insan yapar.",
    )


def triage(message: IncomingMessage, assistant: Assistant) -> TriageResult:
    """Classify an incoming message, draft an answer, and decide who may send it."""
    intent = classify_intent(message.text)
    mode = MODE_FOR_INTENT[intent]
    answer = assistant.ask(message.text, mode=mode)
    decision, reason = decide(message, intent, answer)

    return TriageResult(
        intent=intent,
        # The pipeline may re-route internally (submission rules, channel card),
        # so report the mode that actually produced the answer.
        selected_mode=answer.mode,
        answer=answer,
        evidence_level=answer.evidence,
        source_categories=tuple(scored.chunk.category for scored in answer.retrieved),
        reply_decision=decision,
        reason=reason,
        # Nothing to copy when the archive cannot support an answer.
        draft_reply="" if decision is ReplyDecision.DO_NOT_ANSWER else answer.text,
        message=message,
    )
