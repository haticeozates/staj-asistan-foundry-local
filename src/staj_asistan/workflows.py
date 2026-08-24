"""Rule-based workflows behind the checklist and correction-analyzer modes.

These two modes must be **deterministic**. A participant deciding whether their
submission is complete, or drafting a correction request that a human will act
on, should not depend on sampling temperature. So the structure is computed with
explicit rules and the language model is only allowed to phrase the result.

The rules encode what the instructor actually asked for during the program:
a source-code link (GitHub preferred), a short video (max ~2 minutes) explaining
what you built and what you learned, both sent by e-mail, before September.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import AssistantMode

# --------------------------------------------------------------------------------------
# Shared Turkish text helpers
# --------------------------------------------------------------------------------------

_CLAUSE_SPLIT_RE = re.compile(r"[,.;\n]|\bama\b|\bfakat\b|\bancak\b|\bfakat\b", re.IGNORECASE)

#: Turkish first-person negative past tense (çekmedim, yapmadım, atmadım...) plus
#: the standalone negation words that appear in these messages.
_NEGATION_RE = re.compile(
    r"\b\w+m[aeı]dım\b|\b\w+m[aeı]dim\b|\byok\b|\bdeğil\b|\beksik\b|\bhenüz\b|\bolmadı\b|\bhazırlanmadı\b",
    re.IGNORECASE,
)


def _lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]


def _is_negated(clause: str) -> bool:
    return bool(_NEGATION_RE.search(clause))


def detect_state(text: str, keywords: tuple[str, ...]) -> bool | None:
    """Return True (present), False (explicitly absent) or None (not mentioned)."""
    lowered_clauses = [_lower(c) for c in _clauses(text)]
    verdict: bool | None = None
    for clause in lowered_clauses:
        if not any(keyword in clause for keyword in keywords):
            continue
        # An explicit negation anywhere about this item wins over a positive mention.
        if _is_negated(clause):
            return False
        verdict = True
    return verdict


# --------------------------------------------------------------------------------------
# Submission checklist
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChecklistItem:
    key: str
    label: str
    keywords: tuple[str, ...]
    required: bool
    next_step: str


SUBMISSION_ITEMS: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        key="source_code_link",
        label="Kaynak kod linki (GitHub tercih ediliyor)",
        keywords=("github", "repo", "kaynak kod", "source", "gitlab", "kod linki", "public link"),
        required=True,
        next_step="Projeyi GitHub'da public bir repoya yükle ve linkini hazırla.",
    ),
    ChecklistItem(
        key="demo_video",
        label="Kısa demo videosu (~2 dakika: ne yaptım / ne öğrendim)",
        keywords=("video", "demo", "ekran kaydı", "kayıt aldım", "youtube", "loom", "tanıtım"),
        required=True,
        next_step=(
            "2 dakikayı geçmeyen bir video çek: ne yaptığını ve ne öğrendiğini anlat, "
            "sonra videoyu erişilebilir bir yere yükle."
        ),
    ),
    ChecklistItem(
        key="email_submission",
        label="Linklerin e-posta ile eğitmene gönderilmesi",
        keywords=("mail", "e-mail", "eposta", "e-posta", "gönderdim", "ilettim", "attım"),
        required=True,
        next_step="Kod linkini ve video linkini ayrı ayrı tek bir e-postada eğitmene gönder.",
    ),
    ChecklistItem(
        key="readme",
        label="README / proje açıklaması",
        keywords=("readme", "açıklama", "dokümantasyon", "döküman", "belge"),
        required=False,
        next_step="README'ye kurulum adımlarını, mimariyi ve demo sorularını ekle.",
    ),
)


@dataclass
class ChecklistResult:
    ready: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    complete: bool = False

    def as_dict(self) -> dict:
        return {
            "hazir_olanlar": self.ready,
            "eksikler": self.missing,
            "belirsizler": self.unknown,
            "onerilen_adimlar": self.next_steps,
            "teslime_hazir": self.complete,
        }

    def as_text(self) -> str:
        lines: list[str] = []
        lines.append("**Hazır olanlar**")
        lines.extend(f"- {item}" for item in self.ready) if self.ready else lines.append(
            "- (henüz yok)"
        )
        lines.append("")
        lines.append("**Eksikler**")
        lines.extend(f"- {item}" for item in self.missing) if self.missing else lines.append(
            "- Zorunlu bir eksik görünmüyor."
        )
        if self.unknown:
            lines.append("")
            lines.append("**Mesajından anlaşılmayanlar**")
            lines.extend(f"- {item}" for item in self.unknown)
        lines.append("")
        lines.append("**Önerilen sonraki adımlar**")
        lines.extend(f"{i}. {step}" for i, step in enumerate(self.next_steps, start=1))
        return "\n".join(lines)


def analyze_submission(status_text: str) -> ChecklistResult:
    """Turn a free-text status report into a ready/missing checklist."""
    result = ChecklistResult()
    for item in SUBMISSION_ITEMS:
        state = detect_state(status_text, item.keywords)
        if state is True:
            result.ready.append(item.label)
        elif state is False:
            result.missing.append(item.label)
            result.next_steps.append(item.next_step)
        elif item.required:
            result.missing.append(f"{item.label} — mesajında belirtilmemiş")
            result.next_steps.append(item.next_step)
        else:
            result.unknown.append(item.label)

    required_missing = [
        item
        for item in SUBMISSION_ITEMS
        if item.required and detect_state(status_text, item.keywords) is not True
    ]
    result.complete = not required_missing
    if result.complete:
        result.next_steps.append(
            "Teslim tamam görünüyor: e-postanı gönderdiysen listede satırının güncellenmesini bekle."
        )
    else:
        result.next_steps.append("Eksikleri tamamladıktan sonra tüm linkleri tek e-postada gönder.")
    return result


_REQUIREMENT_QUERY_HINTS = (
    "ne gerekiyor",
    "neler gerekiyor",
    "ne gerekli",
    "neler gerekli",
    "ne lazım",
    "zorunlu teslim",
    "tesliminde",
    "final teslim",
)
_STATUS_QUERY_HINTS = (
    "çekmedim",
    "eksiğim",
    "hazır mıyım",
    "repo hazır",
    "attım",
    "gönderdim",
    "teslim durumu",
)

SUBMISSION_RULE_LINES: tuple[str, ...] = (
    "GitHub / kaynak kod linki gerekir (public bir link de kabul edilir).",
    "Kısa demo videosu gerekir: ne yaptım, ne öğrendim (yaklaşık 2 dakika).",
    "Kod ve video linkleri e-posta ile gönderilmelidir.",
    "Video yoksa teslim eksik sayılır.",
    "WhatsApp mesajı tek başına teslim yerine geçmez.",
)


def is_submission_requirement_query(text: str) -> bool:
    """True when the user asks what the submission rules are, not about their own status."""
    lowered = _lower(text)
    if any(hint in lowered for hint in _STATUS_QUERY_HINTS):
        return False
    if "teslim" not in lowered and "final teslim" not in lowered:
        return False
    return any(hint in lowered for hint in _REQUIREMENT_QUERY_HINTS)


def format_submission_rules() -> str:
    """Canonical submission-requirement card used for 'what do I need to submit?' questions."""
    lines = ["**Final teslim için gerekenler**"]
    lines.extend(f"- {line}" for line in SUBMISSION_RULE_LINES)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Correction request analyzer
# --------------------------------------------------------------------------------------

#: Project options announced by the instructor.
PROJECT_CATALOG: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Building Your First Local RAG Application with Foundry Local",
        ("foundry", "rag", "local rag", "foundry local"),
    ),
    ("Stock Price Prediction with PyTorch", ("pytorch", "stock", "hisse", "fiyat tahmin")),
    ("Quantum Kickstart with Q#", ("quantum", "kuantum", "q#", "qsharp")),
    ("Titanic Survival Analysis", ("titanic",)),
    ("Kendi projem", ("kendi projem", "kendi proje")),
)

LANGUAGE_OPTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Türkçe", ("türkçe", "turkce", "turkish")),
    ("English", ("ingilizce", "english", "ingilzce")),
)

_EMAIL_PRESENT_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Written with explicit case variants rather than re.IGNORECASE: Turkish dotted and
# dotless "i" do not fold reliably under Python's case-insensitive matching.
_NAME_PATTERN_RE = re.compile(
    r"\b(?:[Aa]d[ıi]m|[Aa]d[ıi]\s*[Ss]oyad[ıi]m|[İIi]smim|[Bb]en)\s*[:,]?\s+"
    r"([A-ZÇĞİÖŞÜ][\w'’]+(?:\s+[A-ZÇĞİÖŞÜ][\w'’]+){1,2})"
)
_NAME_LABEL_RE = re.compile(
    r"\b[Aa]d\s*[- ]?\s*[Ss]oyad\s*[:]\s*([A-ZÇĞİÖŞÜ][\w'’]+(?:\s+[A-ZÇĞİÖŞÜ][\w'’]+){1,2})"
)
_STANDALONE_NAME_RE = re.compile(
    r"(?:^|\n)\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü'’]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü'’]+){1,2})\s*(?:$|\n)"
)

_LIST_KEYWORDS = ("liste", "satır", "kayıt", "tablo", "dosyada", "görünüyor", "gözüküyor")
_NAME_CORRECTION_KEYWORDS = ("ad", "isim", "soyad", "yazım")


def _detect_project(text: str) -> str | None:
    lowered = _lower(text)
    for canonical, keywords in PROJECT_CATALOG:
        if any(keyword in lowered for keyword in keywords):
            return canonical
    return None


def _detect_language(text: str) -> str | None:
    lowered = _lower(text)
    for canonical, keywords in LANGUAGE_OPTIONS:
        if any(keyword in lowered for keyword in keywords):
            return canonical
    return None


def _detect_status(text: str) -> str | None:
    lowered = _lower(text)
    if re.search(r"devam etmek istemiyorum|devam etmeyeceğim|bırakmak istiyorum", lowered):
        return "Devam etmek istemiyorum"
    if re.search(r"devam etmek istiyorum|devam edeceğim|devam ediyorum", lowered):
        return "Devam etmek istiyorum"
    return None


def _detect_name(text: str) -> str | None:
    for pattern in (_NAME_LABEL_RE, _NAME_PATTERN_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    standalone = _STANDALONE_NAME_RE.search(text)
    if standalone:
        candidate = standalone.group(1).strip()
        if _lower(candidate).split()[0] not in {"listede", "projem", "merhaba", "teşekkürler"}:
            return candidate
    return None


def _detect_intent(text: str, fields: dict) -> str:
    lowered = _lower(text)
    if any(keyword in lowered for keyword in _LIST_KEYWORDS):
        return "list_correction"
    if fields.get("project"):
        return "project_change"
    if fields.get("language"):
        return "language_change"
    if fields.get("status"):
        return "participation_status"
    if any(keyword in lowered for keyword in _NAME_CORRECTION_KEYWORDS):
        return "name_correction"
    return "unknown"


@dataclass
class CorrectionAnalysis:
    intent: str
    detected_fields: dict
    missing_required_info: list[str]
    recommendation: str
    confidence: str
    suggested_email: str

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "detected_fields": self.detected_fields,
            "missing_required_info": self.missing_required_info,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            # The assistant never edits the roster; a human applies the change.
            "auto_apply": False,
        }


def _draft_email(fields: dict, missing: list[str]) -> str:
    name = fields.get("full_name") or "<Ad Soyad>"
    email = "<e-posta adresin>" if "email" in missing else "<kayıtlı e-posta adresin>"
    lines = [
        "Konu: Summer School listesi düzeltme isteği",
        "",
        "Merhaba Hocam,",
        "",
        f"Listedeki kaydımın güncellenmesini rica ediyorum. Ad Soyad: {name}, E-posta: {email}.",
    ]
    if fields.get("project"):
        lines.append(f"Proje: {fields['project']}")
    if fields.get("language"):
        lines.append(f"Dil: {fields['language']}")
    if fields.get("status"):
        lines.append(f"Devam durumu: {fields['status']}")
    lines += ["", "Teşekkür ederim,", name]
    return "\n".join(lines)


def analyze_correction_request(message: str) -> CorrectionAnalysis:
    """Structure a correction message without applying any change."""
    fields: dict = {}
    project = _detect_project(message)
    language = _detect_language(message)
    status = _detect_status(message)
    name = _detect_name(message)
    has_email = bool(_EMAIL_PRESENT_RE.search(message))

    if project:
        fields["project"] = project
    if language:
        fields["language"] = language
    if status:
        fields["status"] = status
    if name:
        fields["full_name"] = name
    if has_email:
        # Presence is recorded; the address itself is never stored or indexed.
        fields["email"] = "mesajda belirtilmiş"

    intent = _detect_intent(message, fields)

    missing: list[str] = []
    if not name:
        missing.append("full_name")
    if not has_email:
        missing.append("email")
    if intent in {"list_correction", "unknown"} and not (project or language or status):
        missing.append("requested_change")

    if missing:
        recommendation = (
            "Düzeltme isteğini eğitmene e-posta olarak gönder ve eksik bilgileri ekle: "
            + ", ".join(missing)
            + ". WhatsApp mesajı tek başına yeterli değil; eğitmen listedeki doğru satırı "
            "ancak ad-soyad ve e-posta ile eşleştirebiliyor."
        )
        confidence = "low" if len(missing) > 1 else "medium"
    else:
        recommendation = (
            "İstek net görünüyor. Aynı bilgileri e-posta olarak eğitmene gönder; "
            "düzeltmeler toplu olarak ve insan kontrolüyle uygulanıyor."
        )
        confidence = "high"

    return CorrectionAnalysis(
        intent=intent,
        detected_fields=fields,
        missing_required_info=missing,
        recommendation=recommendation,
        confidence=confidence,
        suggested_email=_draft_email(fields, missing),
    )


# --------------------------------------------------------------------------------------
# Mode hinting
# --------------------------------------------------------------------------------------

_MODE_HINTS: tuple[tuple[AssistantMode, tuple[str, ...]], ...] = (
    (
        AssistantMode.CORRECTION_ANALYZER,
        ("listede", "satırım", "yanlış görünüyor", "düzeltme", "güncellenmesini", "devam etmek"),
    ),
    (
        AssistantMode.SUBMISSION_CHECKLIST,
        (
            "eksiğim",
            "teslim için",
            "hazır mıyım",
            "çekmedim",
            "repo hazır",
            "teslim durumu",
            "tesliminde",
            "final teslim",
        ),
    ),
    (
        AssistantMode.TECHNICAL_HELP,
        ("hata", "error", "çalışmıyor", "kurulum", "python", "cuda", "gpu", "endpoint", "port"),
    ),
)


def suggest_mode(text: str) -> AssistantMode:
    """Best-guess mode for a given input, used to nudge the user in the UI."""
    lowered = _lower(text)
    for mode, keywords in _MODE_HINTS:
        if any(keyword in lowered for keyword in keywords):
            return mode
    return AssistantMode.INSTRUCTOR_QA
