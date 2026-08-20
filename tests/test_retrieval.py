"""Retrieval tests: correct sources, instructor authority, and refusal to guess."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from staj_asistan.embeddings import HashingEmbeddings, resolve_embedding_backend
from staj_asistan.models import AuthorRole, Chunk
from staj_asistan.retriever import BM25Index, RetrievalConfig, Retriever
from staj_asistan.vector_store import NumpyVectorStore, build_vector_store


def chunk(chunk_id, text, *, instructor=False, day=1, source="grup1"):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        role=AuthorRole.INSTRUCTOR if instructor else AuthorRole.PARTICIPANT,
        senders=("Barbaros Günay Microsoft",) if instructor else ("Katılımcı#0001",),
        start_time=datetime(2026, 7, day, 10, 0),
        instructor_ratio=1.0 if instructor else 0.0,
    )


CORPUS = [
    chunk(
        "c1",
        "Barbaros Günay Microsoft: Final teslimi için bana kaynak kodun olduğu bir link "
        "(github mesela) ve ne yaptığınızı anlatan kısa bir video linki e-mail atın.",
        instructor=True,
        day=24,
    ),
    chunk(
        "c2",
        "Barbaros Günay Microsoft: Sertifika tek, projeyi bitiren herkese Ağustos ortasına "
        "kadar sertifika gönderiyorum.",
        instructor=True,
        day=27,
    ),
    chunk(
        "c3",
        "Barbaros Günay Microsoft: Foundry Local disconnected bir ortamda çözüm yaratma "
        "deneyimi sunuyor, küçük modeller birçok problemde yeterli olabiliyor.",
        instructor=True,
        day=13,
    ),
    chunk(
        "c4",
        "Katılımcı#0001: Bence sertifika gelmez, ben duymadım böyle bir şey.",
        day=20,
    ),
    chunk(
        "c5",
        "Katılımcı#0002: Kickoff toplantısı çok güzeldi, Python kurulumunu yeni bitirdim.",
        day=5,
    ),
]


def build_retriever(chunks=CORPUS, config=None) -> Retriever:
    backend = HashingEmbeddings()
    store = NumpyVectorStore()
    store.add(chunks, backend.embed_documents([c.text for c in chunks]))
    return Retriever(store, backend, config or RetrievalConfig.calibrated_for(backend.name))


class TestRelevance:
    def test_finds_the_submission_rule(self):
        results = build_retriever().retrieve("Final tesliminde ne gerekiyor?")
        assert results
        assert results[0].chunk.chunk_id == "c1"

    def test_finds_the_certificate_rule(self):
        results = build_retriever().retrieve("sertifika süreci nasıl işliyor")
        assert results[0].chunk.chunk_id == "c2"

    def test_finds_the_foundry_local_rationale(self):
        results = build_retriever().retrieve("Foundry Local neden öneriliyor")
        assert results[0].chunk.chunk_id == "c3"

    def test_results_are_sorted_by_score(self):
        results = build_retriever().retrieve("sertifika")
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)

    def test_every_result_carries_a_score_breakdown(self):
        result = build_retriever().retrieve("video linki")[0]
        assert result.dense_score >= 0
        assert "anlamsal=" in result.explanation


class TestInstructorAuthority:
    def test_instructor_outranks_participant_on_the_same_topic(self):
        results = build_retriever().retrieve("sertifika alacak mıyız")
        top = results[0]
        assert top.chunk.is_instructor
        assert top.chunk.chunk_id == "c2"

    def test_authority_adjustment_has_the_right_sign(self):
        for result in build_retriever().retrieve("sertifika"):
            assert (result.authority_boost > 0) is result.chunk.is_instructor

    def test_participant_guess_does_not_outrank_the_official_answer(self):
        # c4 is a participant claiming "no certificates" and matches the wording of the
        # question almost literally; c2 is the instructor's actual rule. Authority wins.
        results = build_retriever().retrieve("sertifika gelmez mi")
        assert results[0].chunk.chunk_id == "c2"

    def test_authority_weighting_is_what_flips_that_ranking(self):
        neutral = build_retriever(
            config=RetrievalConfig.calibrated_for(
                "hashing", instructor_boost=0.0, participant_penalty=0.0
            )
        )
        assert neutral.retrieve("sertifika gelmez mi")[0].chunk.chunk_id == "c4"

    def test_participant_chunks_are_penalised(self):
        results = build_retriever().retrieve("sertifika")
        participant = [r for r in results if not r.chunk.is_instructor]
        assert all(r.authority_boost < 0 for r in participant)

    def test_curated_documents_are_neither_boosted_nor_penalised(self):
        document = Chunk(
            chunk_id="doc1",
            text="Sertifika katılımcı başına tek veriliyor ve listedeki isme basılıyor.",
            source="notlar",
            role=AuthorRole.DOCUMENT,
            start_time=datetime(2026, 7, 20, 10, 0),
        )
        results = build_retriever(CORPUS + [document]).retrieve("sertifika")
        found = [r for r in results if r.chunk.chunk_id == "doc1"]
        assert found and found[0].authority_boost == 0.0

    def test_instructor_only_filter(self):
        results = build_retriever().retrieve("sertifika", instructor_only=True)
        assert results
        assert all(r.chunk.is_instructor for r in results)

    def test_threshold_is_calibrated_per_backend(self):
        assert RetrievalConfig.calibrated_for("hashing").min_score < RetrievalConfig.calibrated_for(
            "sentence-transformers"
        ).min_score


class TestRefusalPath:
    def test_unrelated_question_returns_nothing(self):
        results = build_retriever().retrieve("İstanbul'da hava durumu nasıl olacak yarın")
        assert results == []

    def test_empty_query_returns_nothing(self):
        assert build_retriever().retrieve("   ") == []

    def test_empty_index_returns_nothing(self):
        backend = HashingEmbeddings()
        retriever = Retriever(NumpyVectorStore(), backend)
        assert retriever.retrieve("herhangi bir soru") == []

    def test_threshold_is_configurable(self):
        strict = build_retriever(config=RetrievalConfig.calibrated_for("hashing", min_score=0.99))
        assert strict.retrieve("Final tesliminde ne gerekiyor?") == []

    def test_authoritative_but_irrelevant_chunk_stays_below_threshold(self):
        # Regression guard: boosts must scale relevance, never add a constant floor.
        instructor_only_corpus = [c for c in CORPUS if c.is_instructor]
        results = build_retriever(instructor_only_corpus).retrieve("yarın hava nasıl olacak")
        assert results == []


class TestDiversity:
    def test_near_duplicate_reposts_are_collapsed(self):
        duplicated = [
            chunk(f"d{i}", CORPUS[0].text, instructor=True, day=i + 1) for i in range(5)
        ] + [CORPUS[2]]
        results = build_retriever(duplicated).retrieve("teslim için link ve video", top_k=3)
        texts = [r.chunk.text for r in results]
        assert len(set(texts)) == len(texts) or len(results) <= 2

    def test_top_k_is_respected(self):
        assert len(build_retriever().retrieve("sertifika video github", top_k=2)) <= 2


class TestSourceFiltering:
    def test_can_restrict_to_one_source(self):
        mixed = CORPUS + [chunk("c9", "Sertifika ne zaman gelir acaba", source="grup2", day=25)]
        results = build_retriever(mixed).retrieve("sertifika", sources={"grup2"})
        assert all(r.chunk.source == "grup2" for r in results)


class TestBM25:
    def test_scores_the_matching_document_highest(self):
        index = BM25Index(CORPUS)
        scores = index.scores("sertifika")
        assert scores[1] == max(scores)

    def test_stopwords_do_not_match(self):
        index = BM25Index(CORPUS)
        assert max(index.scores("bir ve veya çok")) == 0.0


class TestVectorStore:
    def test_roundtrip_persistence(self, tmp_path):
        backend = HashingEmbeddings()
        store = NumpyVectorStore()
        store.add(CORPUS, backend.embed_documents([c.text for c in CORPUS]))
        store.save(tmp_path / "index")

        restored = NumpyVectorStore.load(tmp_path / "index")
        assert len(restored) == len(CORPUS)
        assert restored.all_chunks()[0].chunk_id == "c1"
        assert restored.all_chunks()[0].start_time == CORPUS[0].start_time
        assert restored.all_chunks()[0].role is AuthorRole.INSTRUCTOR

    def test_mismatched_counts_raise(self):
        store = NumpyVectorStore()
        with pytest.raises(ValueError):
            store.add(CORPUS, np.zeros((2, 8), dtype=np.float32))

    def test_dimension_change_raises(self):
        store = NumpyVectorStore()
        store.add(CORPUS[:1], np.zeros((1, 8), dtype=np.float32))
        with pytest.raises(ValueError):
            store.add(CORPUS[1:2], np.zeros((1, 16), dtype=np.float32))

    def test_clear_empties_the_index(self):
        store = NumpyVectorStore()
        store.add(CORPUS[:1], np.zeros((1, 8), dtype=np.float32))
        store.clear()
        assert len(store) == 0

    def test_factory_builds_numpy_backend(self):
        assert isinstance(build_vector_store("numpy"), NumpyVectorStore)

    def test_factory_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            build_vector_store("pinecone")


class TestEmbeddingBackends:
    def test_hashing_vectors_are_normalised(self):
        vectors = HashingEmbeddings().embed_documents(["merhaba dünya", "final teslimi"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_hashing_is_deterministic(self):
        first = HashingEmbeddings().embed_query("sertifika ne zaman")
        second = HashingEmbeddings().embed_query("sertifika ne zaman")
        assert np.array_equal(first, second)

    def test_similar_texts_score_higher_than_unrelated(self):
        backend = HashingEmbeddings()
        base = backend.embed_query("final teslimi için github linki")
        similar = backend.embed_query("teslim için github linki lazım")
        unrelated = backend.embed_query("kuantum bilgisayarlar ve süperpozisyon")
        assert float(base @ similar) > float(base @ unrelated)

    def test_offline_resolution_picks_hashing(self, monkeypatch):
        monkeypatch.setenv("STAJ_ASISTAN_OFFLINE", "1")
        monkeypatch.setenv("STAJ_ASISTAN_EMBEDDING_BACKEND", "auto")
        assert resolve_embedding_backend().name == "hashing"

    def test_unknown_backend_name_raises(self):
        with pytest.raises(ValueError):
            resolve_embedding_backend("word2vec")
