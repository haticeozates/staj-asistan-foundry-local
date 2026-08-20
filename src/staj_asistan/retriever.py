"""Hybrid retrieval with authority weighting.

Pure dense retrieval underperforms on this corpus: questions contain exact
program vocabulary ("sertifika", "video", "GitHub linki") that lexical matching
nails, while paraphrased questions need semantics. So the score combines both,
then adds two domain signals:

* **authority** - the instructor is the source of truth; a participant guessing
  in the group is not. Instructor-dominated chunks get boosted.
* **recency** - program rules were refined over the summer; later announcements
  supersede earlier ones.

Finally MMR trims near-duplicates, which matters because the instructor reposts
the same announcement almost verbatim every couple of weeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from .embeddings import EmbeddingBackend, turkish_tokens
from .models import AuthorRole, Chunk, ScoredChunk
from .vector_store import VectorStore

#: Very common Turkish words carry no retrieval signal.
TURKISH_STOPWORDS = frozenset(
    """
    acaba ama ancak artik asla aslinda az bana bazi bazen belki ben benim beri bile bir biraz
    birçok birkaç birşey biz bizim bu buna bunda bundan bunu bunun burada çok çünkü da daha de
    defa değil diye eğer en gibi hem hep hepsi her herhangi hiç için ile ise kez ki kim mi mı mu
    mü nasıl ne neden nerde nerede nereye niçin niye o olarak olduğu oldu olur onlar ona ondan onu
    onun oysa öyle sanki şey siz sizin şu şuna şunda şundan şunu tüm ve veya ya yani var yok
    """.split()
)


#: Cosine similarity lives on a different scale for every embedding model, and the
#: lexical coverage guard only earns its keep on the non-semantic fallback. Both are
#: therefore calibrated per backend, from measurements on the sample corpus rather
#: than from guesswork (see docs/architecture.md).
BACKEND_CALIBRATION: dict[str, dict[str, float]] = {
    # Bag-of-words vectors: scores are small, so the floor is low and the lexical
    # guard does the heavy lifting for off-topic questions.
    "hashing": {"min_score": 0.10, "max_df_ratio": 0.5},
    # Real sentence embeddings separate on-topic from off-topic by score alone.
    "sentence-transformers": {"min_score": 0.30, "max_df_ratio": 0.5},
    "foundry-local": {"min_score": 0.30, "max_df_ratio": 0.5},
}
FALLBACK_CALIBRATION = {"min_score": 0.20, "max_df_ratio": 0.5}


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 6
    candidate_pool: int = 30
    dense_weight: float = 0.65
    lexical_weight: float = 0.35
    #: Multiplicative bonus for instructor-authored content.
    instructor_boost: float = 0.25
    #: Multiplicative penalty for chunks with no instructor content. Participant-only
    #: messages are context and opinion, not program rules.
    participant_penalty: float = 0.5
    recency_weight: float = 0.10
    recency_half_life_days: float = 45.0
    mmr_lambda: float = 0.72
    #: Chunks this similar to an already-selected one are reposts; drop them.
    duplicate_threshold: float = 0.85
    #: Refuse the question unless it shares a term appearing in at most this fraction
    #: of documents. Set to 1.0 to disable the guard.
    max_df_ratio: float = FALLBACK_CALIBRATION["max_df_ratio"]
    #: Below this final score the pipeline refuses to answer instead of guessing.
    min_score: float = FALLBACK_CALIBRATION["min_score"]

    @classmethod
    def calibrated_for(cls, backend_name: str, **overrides) -> RetrievalConfig:
        """Build a config whose thresholds match the active embedding backend."""
        calibration = dict(BACKEND_CALIBRATION.get(backend_name, FALLBACK_CALIBRATION))
        calibration.update(overrides)
        return cls(**calibration)


DEFAULT_RETRIEVAL = RetrievalConfig()


#: Fixed-length truncation ("F5") is a well-known strong baseline stemmer for Turkish:
#: the language is agglutinative, so the first few characters carry the stem while
#: suffixes vary ("teslim", "teslimi", "tesliminde" -> "tesli").
STEM_LENGTH = 5


def turkish_stem(token: str) -> str:
    return token[:STEM_LENGTH]


def content_tokens(text: str) -> list[str]:
    """Stopword-filtered, stemmed tokens used for lexical scoring."""
    return [
        turkish_stem(t)
        for t in turkish_tokens(text)
        if len(t) > 2 and t not in TURKISH_STOPWORDS
    ]


class BM25Index:
    """Compact BM25-Okapi implementation over chunk texts."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens = [content_tokens(c.text) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for tokens in self.doc_tokens:
            freqs: dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.term_freqs.append(freqs)
            for token in freqs:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        total = len(chunks) or 1
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()
        }
        self.doc_freq = doc_freq
        self.total_docs = total

    def has_discriminative_overlap(self, query_stems: set[str], max_df_ratio: float) -> bool:
        """True when the question shares at least one *distinctive* word with the corpus.

        A question about pizza or football shares nothing with program announcements
        and must be refused. A question like "Mars'a ne zaman insan gönderilecek?" is
        trickier: "zaman" and "gönder" do occur in the corpus, but they occur nearly
        everywhere, so they carry no topical signal. Requiring an overlap on a term
        that appears in at most ``max_df_ratio`` of the documents captures that
        distinction with one cheap check and without a tuned similarity floor.
        """
        if not query_stems or not self.doc_freq:
            return True
        limit = max(self.total_docs * max_df_ratio, 1.0)
        return any(0 < self.doc_freq.get(stem, 0) <= limit for stem in query_stems)

    def scores(self, query: str) -> list[float]:
        query_tokens = content_tokens(query)
        out = [0.0] * len(self.chunks)
        if not query_tokens or not self.avg_len:
            return out
        for index, freqs in enumerate(self.term_freqs):
            length = self.doc_len[index] or 1
            total = 0.0
            for token in query_tokens:
                freq = freqs.get(token)
                if not freq:
                    continue
                denominator = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf.get(token, 0.0) * freq * (self.k1 + 1) / denominator
            out[index] = total
        return out


def _saturate(value: float, midpoint: float = 8.0) -> float:
    """Map an unbounded positive score into ``[0, 1)`` without corpus normalisation."""
    return value / (value + midpoint) if value > 0 else 0.0


def _recency_factor(chunk: Chunk, reference: datetime | None, half_life_days: float) -> float:
    if chunk.start_time is None or reference is None:
        return 0.0
    age_days = max((reference - chunk.start_time).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / half_life_days)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Retriever:
    """Combines a vector store with lexical scoring and domain-specific boosts."""

    def __init__(
        self,
        store: VectorStore,
        backend: EmbeddingBackend,
        config: RetrievalConfig = DEFAULT_RETRIEVAL,
    ) -> None:
        self.store = store
        self.backend = backend
        self.config = config
        self._bm25: BM25Index | None = None
        self._bm25_size = -1
        self._latest: datetime | None = None

    def _ensure_lexical_index(self) -> BM25Index:
        if self._bm25 is None or self._bm25_size != len(self.store):
            chunks = self.store.all_chunks()
            self._bm25 = BM25Index(chunks)
            self._bm25_size = len(chunks)
            timestamps = [c.start_time for c in chunks if c.start_time]
            self._latest = max(timestamps) if timestamps else None
        return self._bm25

    def invalidate(self) -> None:
        self._bm25 = None
        self._bm25_size = -1

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        instructor_only: bool = False,
        sources: set[str] | None = None,
    ) -> list[ScoredChunk]:
        config = self.config
        limit = top_k or config.top_k
        if len(self.store) == 0 or not query.strip():
            return []

        lexical = self._ensure_lexical_index()
        chunks = lexical.chunks
        index_of = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        query_stems = set(content_tokens(query))
        if not lexical.has_discriminative_overlap(query_stems, config.max_df_ratio):
            return []

        dense_hits = self.store.search(
            self.backend.embed_query(query), min(config.candidate_pool, len(chunks))
        )
        lexical_scores = lexical.scores(query)
        lexical_ranked = sorted(range(len(chunks)), key=lambda i: -lexical_scores[i])[
            : config.candidate_pool
        ]

        candidates: dict[str, ScoredChunk] = {}

        def register(chunk: Chunk, dense: float) -> None:
            position = index_of.get(chunk.chunk_id)
            lexical_value = _saturate(lexical_scores[position]) if position is not None else 0.0
            dense_value = max(dense, 0.0)
            relevance = config.dense_weight * dense_value + config.lexical_weight * lexical_value
            # Boosts scale relevance instead of adding to it, so an authoritative but
            # irrelevant chunk stays below the refusal threshold.
            if chunk.role is AuthorRole.DOCUMENT:
                # Curated notes are neither authoritative nor hearsay: leave them as-is.
                authority_multiplier = 1.0
            else:
                ratio = chunk.instructor_ratio
                authority_multiplier = (1 + config.instructor_boost * ratio) * (
                    1 - config.participant_penalty * (1 - ratio)
                )
            recency_multiplier = 1 + config.recency_weight * _recency_factor(
                chunk, self._latest, config.recency_half_life_days
            )
            candidates[chunk.chunk_id] = ScoredChunk(
                chunk=chunk,
                score=relevance * authority_multiplier * recency_multiplier,
                dense_score=dense_value,
                lexical_score=lexical_value,
                authority_boost=relevance * recency_multiplier * (authority_multiplier - 1),
                recency_boost=relevance * authority_multiplier * (recency_multiplier - 1),
            )

        for chunk, dense in dense_hits:
            register(chunk, dense)
        for position in lexical_ranked:
            if lexical_scores[position] <= 0:
                continue
            chunk = chunks[position]
            if chunk.chunk_id not in candidates:
                register(chunk, 0.0)

        pool = list(candidates.values())
        if instructor_only:
            pool = [s for s in pool if s.chunk.is_instructor]
        if sources:
            pool = [s for s in pool if s.chunk.source in sources]
        pool = [s for s in pool if s.score >= config.min_score]
        pool.sort(key=lambda s: -s.score)
        return self._diversify(pool, limit)

    def _diversify(self, pool: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        """Drop reposted duplicates, then apply MMR for topical diversity."""
        if len(pool) <= 1:
            return pool[:limit]
        lam = self.config.mmr_lambda
        threshold = self.config.duplicate_threshold
        token_sets = {s.chunk.chunk_id: set(content_tokens(s.chunk.text)) for s in pool}
        selected: list[ScoredChunk] = [pool[0]]
        remaining = pool[1:]

        while remaining and len(selected) < limit:
            best_index, best_value = None, -math.inf
            for index, candidate in enumerate(remaining):
                similarity = max(
                    _jaccard(token_sets[candidate.chunk.chunk_id], token_sets[s.chunk.chunk_id])
                    for s in selected
                )
                if similarity >= threshold:
                    continue
                value = lam * candidate.score - (1 - lam) * similarity
                if value > best_value:
                    best_index, best_value = index, value
            if best_index is None:
                break
            selected.append(remaining.pop(best_index))
        return selected
