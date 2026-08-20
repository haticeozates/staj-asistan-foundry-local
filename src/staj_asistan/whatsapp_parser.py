"""Parser for WhatsApp chat exports (iOS and Android, Turkish and English locales).

Real exports are messier than the format suggests:

* iOS wraps senders and phone numbers in Unicode bidi isolates
* long announcements span dozens of lines, including tab-separated tables
* joins, security-code notices and "media omitted" markers are noise
* the same instructor appears under several spellings — full name with the organisation
  suffix, an ASCII-folded variant, or just a first name — depending on who saved the contact

Everything here is deterministic and unit-tested; no model is involved.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from .models import AuthorRole, Message
from .privacy import (
    DEFAULT_POLICY,
    MaskingReport,
    PrivacyPolicy,
    mask_text,
    pseudonymize_sender,
    strip_bidi,
)

_MERIDIEM = {"am": 0, "pm": 12, "öö": 0, "ös": 12, "oo": 0, "os": 12}

_IOS_LINE_RE = re.compile(
    r"^\[(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*"
    r"(?P<meridiem>AM|PM|ÖÖ|ÖS|ÖÖ\.|ÖS\.)?\]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_ANDROID_LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*"
    r"(?P<meridiem>AM|PM|ÖÖ|ÖS)?\s+-\s+(?P<rest>.+)$",
    re.IGNORECASE,
)

_EDITED_SUFFIX_RE = re.compile(
    r"\s*<\s*(?:Bu mesaj düzenlendi|This message was edited)\s*>\s*$", re.IGNORECASE
)

_SYSTEM_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"güvenlik kodu değişti",
        r"security code changed",
        r"bir grup bağlantısıyla katıldı",
        r"joined using this group's invite link",
        r"kişisini ekledi",
        r"gruba ekledi",
        r"added you",
        r"gruptan ayrıldı",
        r"left$",
        r"çıkardı",
        r"removed",
        r"grubu oluşturdu",
        r"created group",
        r"grubun konusunu değiştirdi",
        r"grup açıklamasını",
        r"grubun simgesini",
        r"changed the subject",
        r"changed this group's icon",
        r"telefon numarasını değiştirdi",
        r"changed their phone number",
        r"uçtan uca şifrelidir",
        r"end-to-end encrypted",
        r"dahil edilmedi",  # görüntü/ses/belge/sticker dahil edilmedi
        r"omitted>?$",
        r"bu mesaj silindi",
        r"this message was deleted",
        r"mesajı sildiniz",
        r"null$",
    )
)


def _fold(text: str) -> str:
    """Lowercase and strip Turkish diacritics so name variants compare equal."""
    text = strip_bidi(text).replace("ı", "i").replace("İ", "i")
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_only).strip().lower()


#: Deliberately generic. The real instructor's name belongs to a person, so it is supplied
#: per cohort through ``STAJ_ASISTAN_INSTRUCTOR_ALIASES`` rather than shipped in the repo.
_DEFAULT_INSTRUCTOR_ALIASES = (
    "eğitmen",
    "egitmen",
    "instructor",
)


def _instructor_aliases_from_env() -> tuple[str, ...]:
    raw = os.getenv("STAJ_ASISTAN_INSTRUCTOR_ALIASES", "")
    custom = tuple(alias.strip() for alias in raw.split(",") if alias.strip())
    return custom or _DEFAULT_INSTRUCTOR_ALIASES


@dataclass(frozen=True)
class InstructorIdentity:
    """Recognises the program instructor across saved-contact variants.

    Aliases are the cohort's configuration, not a constant: set
    ``STAJ_ASISTAN_INSTRUCTOR_ALIASES="ad soyad,ad soyad kurum"`` for another program.
    Mailbox names and phone numbers are deliberately not defaults — they belong to a
    person and this list ships publicly.
    """

    aliases: tuple[str, ...] = field(default_factory=lambda: _instructor_aliases_from_env())

    def matches(self, sender: str) -> bool:
        folded = _fold(sender)
        if not folded:
            return False
        return any(alias in folded for alias in (_fold(a) for a in self.aliases))


DEFAULT_INSTRUCTOR = InstructorIdentity()


def _parse_timestamp(date_str: str, time_str: str, meridiem: str | None) -> datetime | None:
    parts = re.split(r"[./-]", date_str)
    if len(parts) != 3:
        return None
    try:
        first, second, year_part = (int(p) for p in parts)
    except ValueError:
        return None
    year = year_part if year_part > 100 else 2000 + year_part

    time_parts = [int(p) for p in time_str.split(":")]
    hour, minute = time_parts[0], time_parts[1]
    second_of_minute = time_parts[2] if len(time_parts) > 2 else 0
    if meridiem:
        base = _MERIDIEM.get(_fold(meridiem).rstrip("."), 0)
        hour = (hour % 12) + base

    # Turkish exports are day-first; fall back to month-first for US-style exports.
    for day, month in ((first, second), (second, first)):
        try:
            return datetime(year, month, day, hour, minute, second_of_minute)
        except ValueError:
            continue
    return None


def _is_system_event(sender: str, text: str) -> bool:
    if not sender:
        return True
    body = text.strip()
    return any(p.search(body) for p in _SYSTEM_PATTERNS)


def _split_sender(rest: str) -> tuple[str, str]:
    """Split ``Sender: body`` while tolerating senders that are phone numbers."""
    if ":" not in rest:
        return "", rest.strip()
    sender, body = rest.split(":", 1)
    if len(sender) > 80 or "\n" in sender:
        return "", rest.strip()
    return sender.strip(), body.strip()


def looks_like_whatsapp_export(text: str) -> bool:
    """Heuristic used by ingestion to pick the right parser for a file.

    Uses a ratio as well as a count so that a very short export (a handful of
    messages) is still recognised, while a prose document that happens to quote
    one timestamped line is not.
    """
    head = [line for line in strip_bidi(text).splitlines()[:60] if line.strip()]
    if not head:
        return False
    hits = sum(bool(_IOS_LINE_RE.match(line) or _ANDROID_LINE_RE.match(line)) for line in head)
    return hits >= 2 or hits / len(head) >= 0.5


@dataclass
class _RawMessage:
    date_str: str
    time_str: str
    meridiem: str | None
    sender: str
    body_lines: list[str]
    line_no: int


def parse_whatsapp_export(
    text: str,
    source: str = "whatsapp",
    policy: PrivacyPolicy = DEFAULT_POLICY,
    instructor: InstructorIdentity = DEFAULT_INSTRUCTOR,
    keep_system_events: bool = False,
    report: MaskingReport | None = None,
) -> list[Message]:
    """Parse an export into privacy-masked :class:`Message` objects.

    Pass ``report`` to collect masking statistics from the text that was actually
    masked; counting over the raw file instead would also count chat timestamps.
    """
    seen_senders: set[str] = set()
    raw_messages: list[_RawMessage] = []
    current: _RawMessage | None = None

    for line_no, line in enumerate(strip_bidi(text).replace("\r\n", "\n").split("\n"), start=1):
        match = _IOS_LINE_RE.match(line) or _ANDROID_LINE_RE.match(line)
        if match:
            sender, body = _split_sender(match.group("rest"))
            current = _RawMessage(
                date_str=match.group("date"),
                time_str=match.group("time"),
                meridiem=match.group("meridiem"),
                sender=sender,
                body_lines=[body],
                line_no=line_no,
            )
            raw_messages.append(current)
        elif current is not None:
            current.body_lines.append(line)

    messages: list[Message] = []
    for raw in raw_messages:
        body = _EDITED_SUFFIX_RE.sub("", "\n".join(raw.body_lines)).strip()
        system_event = _is_system_event(raw.sender, body)
        if system_event and not keep_system_events:
            continue

        is_instructor = instructor.matches(raw.sender)
        masked = mask_text(body, policy)
        if not masked.text and not system_event:
            continue

        if report is not None:
            report.merge(masked.report)
            if not is_instructor and raw.sender and raw.sender not in seen_senders:
                seen_senders.add(raw.sender)
                report.counts["sender"] += 1

        display_sender = (
            strip_bidi(raw.sender).strip()
            if is_instructor
            else pseudonymize_sender(raw.sender, policy)
        )
        role = (
            AuthorRole.SYSTEM
            if system_event
            else (AuthorRole.INSTRUCTOR if is_instructor else AuthorRole.PARTICIPANT)
        )
        messages.append(
            Message(
                text=masked.text,
                sender=display_sender,
                role=role,
                source=source,
                timestamp=_parse_timestamp(raw.date_str, raw.time_str, raw.meridiem),
                line_no=raw.line_no,
                is_system_event=system_event,
            )
        )
    return messages


def masking_report_for(text: str, policy: PrivacyPolicy = DEFAULT_POLICY):
    """Total masking statistics for a whole export (used for the sidebar counter)."""
    return mask_text(strip_bidi(text), policy).report
