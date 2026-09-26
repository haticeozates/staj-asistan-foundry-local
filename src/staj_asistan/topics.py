"""Mode-aware topic labels for chunks and questions.

Retrieval is hybrid and can surface a Foundry Local connection-error snippet
for a roster question just because a few stems overlap. Categories let the
pipeline drop those off-lane hits without inventing a second ranker.

Labels are heuristic. Prefer missing a borderline technical note over letting
troubleshooting text answer a "should I keep working?" question.
"""

from __future__ import annotations

from .models import AssistantMode, AuthorRole, ChunkCategory, ScoredChunk

# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


def _fold(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _has_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


# --------------------------------------------------------------------------------------
# Chunk classification
# --------------------------------------------------------------------------------------

#: Connection, runtime and setup troubleshooting — not program rules.
_TECHNICAL_HINTS = (
    "bağlanamıyor",
    "modele bağlan",
    "servisin ayakta",
    "endpoint",
    "base url",
    "execution provider",
    "download_and_register",
    "select_variant",
    "cuda",
    "gpu",
    "cpu üzerinde",
    "python sdk",
    "vektör veritaban",
    "embedding çıkar",
    "local rag mimari",
    "readme'ye kurulum",
    "readme’ye kurulum",
    "nasıl çalıştırılacağını",
    "model yüklü",
    "küçük modellerle başlayın",
    "foundry local kurdum",
    "foundry local çalışm",
)

_ROSTER_HINTS = (
    "isim listesi",
    "listeyi güncel",
    "liste gecikmeli",
    "çalışmaya devam",
    "çalışmayı durdur",
    "satırınızı kontrol",
    "satırınızı",
    "pinli liste",
    "listede adım",
    "listem güncellenmedi",
    "listede sertifika alanı",
    "github linkleri listede",
    "bu liste sadece benim takibim",
)

_CORRECTION_HINTS = (
    "düzeltme istek",
    "düzeltme isteği",
    "düzeltmeler için",
    "düzeltmeleri anında",
    "ad-soyad ve listede",
    "whatsapp grubuna değil",
    "listede kayıtlı e-posta",
    "değiştirilebilen alanlar",
)

_SUBMISSION_HINTS = (
    "final teslim",
    "teslim eksik",
    "kaynak kod link",
    "video link",
    "e-mail atıyorsunuz",
    "e-posta ile gönder",
    "github'a yükleyebilirsiniz",
    "github’a yükleyebilirsiniz",
    "2 dakikada ne yaptığınızı",
    "2 dk max",
    "teslime_hazir",
    "zorunlu teslim",
)

_PROJECT_HINTS = (
    "proje seçenek",
    "stock price prediction",
    "quantum kickstart",
    "titanic survival",
    "kendi projem",
    "disconnected bir ortamda",
    "yapay zekâya ulaşılabilirliği",
)


def classify_chunk(text: str, source: str = "") -> ChunkCategory:
    """Label a chunk from its text, falling back to the source filename."""
    lowered = _fold(text)
    source_l = _fold(source)

    if _has_any(lowered, _TECHNICAL_HINTS):
        return ChunkCategory.TECHNICAL
    if _has_any(lowered, _SUBMISSION_HINTS):
        return ChunkCategory.SUBMISSION
    if _has_any(lowered, _CORRECTION_HINTS):
        return ChunkCategory.CORRECTION
    if _has_any(lowered, _ROSTER_HINTS):
        return ChunkCategory.ROSTER
    if _has_any(lowered, _PROJECT_HINTS):
        return ChunkCategory.PROJECT

    if "submission" in source_l or "teslim" in source_l:
        return ChunkCategory.SUBMISSION
    if "correction" in source_l or "duzeltme" in source_l or "düzeltme" in source_l:
        return ChunkCategory.CORRECTION
    return ChunkCategory.GENERAL


# --------------------------------------------------------------------------------------
# Query intent
# --------------------------------------------------------------------------------------

_ROSTER_QUERY_HINTS = (
    "liste",
    "roster",
    "güncel değil",
    "guncel degil",
    "devam etmeli",
    "çalışmaya devam",
    "calismaya devam",
    "satırım",
    "satirim",
)

_CORRECTION_QUERY_HINTS = (
    "proje yanlış",
    "proje yanlis",
    "düzeltme",
    "duzeltme",
    "adım yanlış",
    "adim yanlis",
    "ismim",
    "güncellenmesini",
    "guncellenmesini",
)

_SUBMISSION_QUERY_HINTS = (
    "teslim",
    "video çek",
    "video cek",
    "github repo",
    "github'a ne",
    "github’a ne",
    "ne koymam",
    "eksiğim",
    "eksigim",
    "kaynak kod",
    "yazmam yeterli",
    "whatsapp'tan yaz",
    "whatsapp’tan yaz",
    "whatsapp yeterli",
)

_TECHNICAL_QUERY_HINTS = (
    "çalışmazsa",
    "calismazsa",
    "çalışmıyor",
    "calismiyor",
    "bağlanamıyor",
    "baglanamiyor",
    "endpoint",
    "gpu",
    "cuda",
    "kurulum",
    "local rag",
)


#: Vocabulary that marks a question as belonging to this programme at all. It exists
#: to stop a coincidental word match from passing as evidence: "İstanbul'da hava
#: nasıl?" can still overlap an indexed chunk lexically, and without this gate the
#: deterministic workflows will happily build a submission checklist out of it.
_PROGRAMME_TOPIC_HINTS = (
    "staj",
    "program",
    "eğitmen",
    "egitmen",
    "sertifika",
    "certificate",
    "proje",
    "video",
    "github",
    "repo",
    "kaynak kod",
    "source code",
    "foundry",
    "local rag",
    "rag",
    "model",
    "python",
    "duyuru",
    "grup",
    "mentor",
    "e-posta",
    "eposta",
    "e-mail",
    "whatsapp",
    # Status phrasing the checklist workflow is built to read. "Her şey hazır" names
    # no programme noun but is unmistakably about a submission.
    "hazır",
    "hazir",
    "bitirdim",
    "tamamladım",
    "tamamladim",
    "çekmedim",
    "cekmedim",
    "yükledim",
    "yukledim",
    "gönderdim",
    "gonderdim",
    "eksik",
)


def is_programme_topic(text: str) -> bool:
    """True when the question is about this programme in any recognisable way.

    Deliberately generous: this is a floor, not a router. Anything it lets through
    is still judged on retrieval evidence, so a false positive costs nothing while
    a false negative would silence a legitimate question.
    """
    lowered = _fold(text)
    return (
        _has_any(lowered, _PROGRAMME_TOPIC_HINTS)
        or _has_any(lowered, _ROSTER_QUERY_HINTS)
        or _has_any(lowered, _CORRECTION_QUERY_HINTS)
        or _has_any(lowered, _SUBMISSION_QUERY_HINTS)
        or _has_any(lowered, _TECHNICAL_QUERY_HINTS)
    )


def infer_query_topic(query: str, mode: AssistantMode) -> ChunkCategory:
    """Map a question (and the active mode) onto the same category vocabulary.

    Explicit workflow modes win. For free-form Instructor Q&A, roster/correction
    phrases beat a coincidental Foundry Local mention.
    """
    if mode is AssistantMode.TECHNICAL_HELP:
        return ChunkCategory.TECHNICAL
    if mode is AssistantMode.SUBMISSION_CHECKLIST:
        return ChunkCategory.SUBMISSION
    if mode is AssistantMode.CORRECTION_ANALYZER:
        return ChunkCategory.CORRECTION

    lowered = _fold(query)
    if _has_any(lowered, _ROSTER_QUERY_HINTS):
        return ChunkCategory.ROSTER
    if _has_any(lowered, _CORRECTION_QUERY_HINTS):
        return ChunkCategory.CORRECTION
    if _has_any(lowered, _SUBMISSION_QUERY_HINTS):
        return ChunkCategory.SUBMISSION
    if _has_any(lowered, _TECHNICAL_QUERY_HINTS):
        return ChunkCategory.TECHNICAL
    return ChunkCategory.GENERAL


#: Categories allowed to answer each query topic. TECHNICAL is excluded from
#: roster/correction/submission so connection-error snippets cannot leak in.
_ALLOWED_CATEGORIES: dict[ChunkCategory, frozenset[ChunkCategory]] = {
    ChunkCategory.ROSTER: frozenset(
        {
            ChunkCategory.ROSTER,
            ChunkCategory.CORRECTION,
            ChunkCategory.SUBMISSION,
            ChunkCategory.GENERAL,
        }
    ),
    ChunkCategory.CORRECTION: frozenset(
        {
            ChunkCategory.CORRECTION,
            ChunkCategory.ROSTER,
            ChunkCategory.SUBMISSION,
        }
    ),
    ChunkCategory.SUBMISSION: frozenset(
        {
            ChunkCategory.SUBMISSION,
            ChunkCategory.CORRECTION,
        }
    ),
    ChunkCategory.TECHNICAL: frozenset(
        {
            ChunkCategory.TECHNICAL,
            ChunkCategory.PROJECT,
            ChunkCategory.GENERAL,
            ChunkCategory.SUBMISSION,
        }
    ),
    ChunkCategory.PROJECT: frozenset(ChunkCategory),
    ChunkCategory.GENERAL: frozenset(ChunkCategory),
}


def allowed_categories(query: str, mode: AssistantMode) -> frozenset[ChunkCategory]:
    return _ALLOWED_CATEGORIES[infer_query_topic(query, mode)]


def _is_participant_question(chunk) -> bool:
    """True for participant-only question turns that should not stand in as rules."""
    if chunk.role is not AuthorRole.PARTICIPANT:
        return False
    text = chunk.text.strip()
    return text.endswith("?") or ("?" in text and "eğitmen:" not in text.lower())


def filter_scored_chunks(
    results: list[ScoredChunk], query: str, mode: AssistantMode
) -> list[ScoredChunk]:
    """Drop off-lane chunks after scoring, keeping original order otherwise."""
    allowed = allowed_categories(query, mode)
    if allowed != frozenset(ChunkCategory):
        results = [item for item in results if item.chunk.category in allowed]
    official = [item for item in results if not _is_participant_question(item.chunk)]
    return official or results
