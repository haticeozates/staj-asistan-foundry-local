"""Privacy layer: strips personal data before anything is indexed or displayed.

Design rule of the project: **raw exports never leave the machine and personal
data never reaches the index**. Masking therefore happens during ingestion, not
at display time, so every downstream artifact (chunks, embeddings, citations,
prompts sent to the model) is already sanitised.

What is removed:

* e-mail addresses (organisational addresses get their own placeholder so the
  assistant can still say "send an e-mail to the instructor" without leaking it)
* phone numbers, including the bidi-wrapped WhatsApp sender format
* private links (Teams meetings, OneDrive, WhatsApp invites, the roster page)
* ``Name <TAB> E-mail`` roster tables that instructors paste into the group
* ``@handle`` mentions
* participant sender names, replaced by a stable salted pseudonym

Deliberate non-goal: general-purpose name detection in free text. Turkish NER is
out of scope for this MVP; see ``docs/privacy.md`` for the residual risk.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace

EMAIL_PLACEHOLDER = "[E-POSTA]"
OFFICIAL_EMAIL_PLACEHOLDER = "[EĞİTMEN E-POSTA]"
PHONE_PLACEHOLDER = "[TELEFON]"
PRIVATE_URL_PLACEHOLDER = "[ÖZEL LİNK]"
HANDLE_PLACEHOLDER = "[KULLANICI]"
ROSTER_PLACEHOLDER = "[KATILIMCI LİSTESİ GİZLENDİ]"

PLACEHOLDERS = (
    EMAIL_PLACEHOLDER,
    OFFICIAL_EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    PRIVATE_URL_PLACEHOLDER,
    HANDLE_PLACEHOLDER,
    ROSTER_PLACEHOLDER,
)


_BASE_PRIVATE_HOSTS = frozenset(
    {
        "teams.microsoft.com",
        "events.teams.microsoft.com",
        "msit.events.teams.microsoft.com",
        "1drv.ms",
        "onedrive.live.com",
        "sharepoint.com",
        "chat.whatsapp.com",
        "wa.me",
    }
)


def _private_hosts_from_env() -> frozenset[str]:
    """Extra hosts to mask, supplied per deployment so none are baked into the repo."""
    raw = os.getenv("STAJ_ASISTAN_PRIVATE_HOSTS", "")
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


@dataclass(frozen=True)
class PrivacyPolicy:
    """Configuration for the masking pass.

    ``official_email_domains`` keeps program-level contact points recognisable
    (as a placeholder) so answers stay actionable, while still never storing the
    address itself.
    """

    mask_emails: bool = True
    mask_phones: bool = True
    mask_private_urls: bool = True
    mask_handles: bool = True
    mask_roster_lines: bool = True
    pseudonymize_senders: bool = True
    official_email_domains: frozenset[str] = frozenset({"microsoft.com"})
    #: RFC 2606 domains reserved for documentation. They cannot belong to a real person or
    #: point at a private resource, so sample files may show a complete request verbatim.
    documentation_domains: frozenset[str] = frozenset(
        {"example.com", "example.org", "example.net", "example.edu"}
    )
    #: Hosts kept verbatim because they are public documentation, not personal data.
    public_url_hosts: frozenset[str] = frozenset(
        {
            "learn.microsoft.com",
            "docs.microsoft.com",
            "github.com",
            "raw.githubusercontent.com",
            "huggingface.co",
            "pypi.org",
            "python.org",
            "streamlit.io",
        }
    )
    #: Hosts always masked, even when they look like a public Microsoft domain.
    #: A cohort's own roster host is deliberately not hardcoded here — naming it in a
    #: public repository would point at the very page this masking protects. Add it
    #: locally with ``STAJ_ASISTAN_PRIVATE_HOSTS=liste.example.net,intranet.example.org``.
    private_url_hosts: frozenset[str] = field(
        default_factory=lambda: _BASE_PRIVATE_HOSTS | _private_hosts_from_env()
    )
    #: ``@mentions`` that are program roles rather than personal handles.
    keep_mentions: frozenset[str] = frozenset(
        {"eğitmen", "egitmen", "instructor", "microsoft", "everyone", "herkes"}
    )
    salt: str = "staj-asistan"


DEFAULT_POLICY = PrivacyPolicy()


@dataclass
class MaskingReport:
    """How much personal data was removed, for the UI and for audits."""

    counts: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: MaskingReport) -> None:
        self.counts.update(other.counts)

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass
class MaskedText:
    text: str
    report: MaskingReport


# WhatsApp wraps phone numbers and handles in bidi isolates; these break every regex.
_BIDI_CHARS = "".join(chr(c) for c in (0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2066, 0x2067, 0x2068, 0x2069, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E))
_BIDI_RE = re.compile(f"[{_BIDI_CHARS}]")

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\"'\]\)]+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ms|co|dev|tr|me)(?:/[^\s<>\"'\]\)]*)?",
    re.IGNORECASE,
)
_HANDLE_RE = re.compile(r"(?<![\w./])@([A-Za-z0-9_.]{3,32})")
_DATE_LIKE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
# Loose candidate: refined by a digit-count check in `_looks_like_phone`.
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s\u00a0.\-]*)?(?:\(?\d{1,4}\)?[\s\u00a0.\-]*){1,6}\d{2,10}")
_TABLE_ROW_RE = re.compile(r"^\s*\"?[^\t]{1,80}\"?\t+\S+@\S+\s*$")
_ANGLE_CONTACT_RE = re.compile(r"[^\s<>]+(?:\s+[^\s<>]+){0,3}\s*<\s*\S+@\S+\s*>")


def strip_bidi(text: str) -> str:
    """Remove Unicode bidi/zero-width control characters used by WhatsApp exports."""
    return _BIDI_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse exotic spacing (NBSP, narrow NBSP) into plain spaces."""
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2007", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _host_of(url: str) -> str:
    host = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    host = host.split("/", 1)[0].split("?", 1)[0].lower()
    return host[4:] if host.startswith("www.") else host


def _is_public_host(host: str, policy: PrivacyPolicy) -> bool:
    if any(host == h or host.endswith("." + h) for h in policy.private_url_hosts):
        return False
    allowed = policy.public_url_hosts | policy.documentation_domains
    return any(host == h or host.endswith("." + h) for h in allowed)


def _looks_like_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 10 <= len(digits) <= 15:
        return False
    # "Son teslim 30.09.2026 23:59" reaches ten digits without containing a phone number.
    if _DATE_LIKE_RE.search(candidate):
        return False
    # A run of digits with no separators and no country code is more likely an id.
    return bool(re.search(r"[\s\u00a0.\-]", candidate) or candidate.startswith("+"))


def _mask_roster_lines(text: str, report: MaskingReport) -> str:
    out: list[str] = []
    previous_was_roster = False
    for line in text.split("\n"):
        if _TABLE_ROW_RE.match(line):
            report.counts["roster_row"] += 1
            if not previous_was_roster:
                out.append(ROSTER_PLACEHOLDER)
            previous_was_roster = True
            continue
        previous_was_roster = False
        out.append(line)
    return "\n".join(out)


def _mask_emails(text: str, policy: PrivacyPolicy, report: MaskingReport) -> str:
    def sub(match: re.Match[str]) -> str:
        domain = match.group(0).rsplit("@", 1)[1].lower()
        if domain in policy.documentation_domains:
            return match.group(0)
        if any(domain == d or domain.endswith("." + d) for d in policy.official_email_domains):
            report.counts["official_email"] += 1
            return OFFICIAL_EMAIL_PLACEHOLDER
        report.counts["email"] += 1
        return EMAIL_PLACEHOLDER

    return _EMAIL_RE.sub(sub, text)


def _mask_urls(text: str, policy: PrivacyPolicy, report: MaskingReport) -> str:
    def sub(match: re.Match[str]) -> str:
        url = match.group(0)
        if _is_public_host(_host_of(url), policy):
            return url
        report.counts["private_url"] += 1
        return PRIVATE_URL_PLACEHOLDER

    return _URL_RE.sub(sub, text)


def _mask_phones(text: str, report: MaskingReport) -> str:
    def sub(match: re.Match[str]) -> str:
        candidate = match.group(0)
        if not _looks_like_phone(candidate):
            return candidate
        report.counts["phone"] += 1
        # Keep any trailing punctuation the loose pattern swallowed.
        trailing = re.search(r"[.\-\s\u00a0]+$", candidate)
        return PHONE_PLACEHOLDER + (trailing.group(0) if trailing else "")

    return _PHONE_RE.sub(sub, text)


def _mask_handles(text: str, policy: PrivacyPolicy, report: MaskingReport) -> str:
    def sub(match: re.Match[str]) -> str:
        handle = match.group(1)
        if handle.lower() in policy.keep_mentions:
            return match.group(0)
        report.counts["handle"] += 1
        return HANDLE_PLACEHOLDER

    return _HANDLE_RE.sub(sub, text)


def _mask_angle_contacts(text: str, report: MaskingReport) -> str:
    def sub(match: re.Match[str]) -> str:
        report.counts["contact_card"] += 1
        return ROSTER_PLACEHOLDER

    return _ANGLE_CONTACT_RE.sub(sub, text)


def mask_text(text: str, policy: PrivacyPolicy = DEFAULT_POLICY) -> MaskedText:
    """Return ``text`` with personal data replaced by stable placeholders."""
    report = MaskingReport()
    cleaned = strip_bidi(text)

    if policy.mask_roster_lines:
        cleaned = _mask_roster_lines(cleaned, report)
        cleaned = _mask_angle_contacts(cleaned, report)
    if policy.mask_emails:
        cleaned = _mask_emails(cleaned, policy, report)
    if policy.mask_private_urls:
        cleaned = _mask_urls(cleaned, policy, report)
    if policy.mask_phones:
        cleaned = _mask_phones(cleaned, report)
    if policy.mask_handles:
        cleaned = _mask_handles(cleaned, policy, report)

    cleaned = "\n".join(normalize_whitespace(line) for line in cleaned.split("\n"))
    return MaskedText(text=cleaned.strip(), report=report)


def pseudonymize_sender(sender: str, policy: PrivacyPolicy = DEFAULT_POLICY) -> str:
    """Map a participant display name to a stable, non-reversible pseudonym.

    The same person keeps the same label across a session (useful for reading a
    conversation) but the label cannot be traced back to the original name.
    """
    if not policy.pseudonymize_senders:
        return sender
    normalized = unicodedata.normalize("NFKC", strip_bidi(sender)).strip().lower()
    digest = hashlib.sha256(f"{policy.salt}:{normalized}".encode()).hexdigest()[:4]
    return f"Katılımcı#{digest}"


def contains_personal_data(text: str, policy: PrivacyPolicy = DEFAULT_POLICY) -> bool:
    """Guard used by tests and by the sample-data generator."""
    return mask_text(text, policy).report.total > 0


def with_extra_private_hosts(policy: PrivacyPolicy, *hosts: str) -> PrivacyPolicy:
    """Convenience helper for deployments that need to mask additional hosts."""
    return replace(policy, private_url_hosts=policy.private_url_hosts | frozenset(hosts))
