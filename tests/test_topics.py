"""Chunk category heuristics and mode-aware retrieval filters."""

from __future__ import annotations

from datetime import datetime

from staj_asistan.models import AssistantMode, AuthorRole, Chunk, ChunkCategory, ScoredChunk
from staj_asistan.topics import (
    classify_chunk,
    filter_scored_chunks,
    infer_query_topic,
)


def _chunk(text: str, source: str = "grup1", category: ChunkCategory | None = None) -> Chunk:
    return Chunk(
        chunk_id="c1",
        text=text,
        source=source,
        role=AuthorRole.INSTRUCTOR,
        start_time=datetime(2026, 7, 6, 11, 20),
        instructor_ratio=1.0,
        category=category or classify_chunk(text, source),
    )


class TestClassifyChunk:
    def test_foundry_connection_error_is_technical(self):
        text = (
            "Foundry Local kurdum ama uygulama modele bağlanamıyor, ne kontrol etmeliyim? "
            "Önce servisin ayakta olduğundan emin olun, sonra hangi portta dinlediğini kontrol edin."
        )
        assert classify_chunk(text) is ChunkCategory.TECHNICAL

    def test_readme_setup_chunk_is_technical(self):
        text = (
            "Projenin GitHub'da düzenli görünmesi için README'ye kurulum adımlarını, "
            "nasıl çalıştırılacağını ve birkaç örnek soruyu koyun. Değerlendirirken ilk oraya bakıyorum."
        )
        assert classify_chunk(text) is ChunkCategory.TECHNICAL

    def test_keep_working_announcement_is_roster(self):
        text = "Düzelecek, hiç problem değil. Liste gecikmeli güncelleniyor, siz çalışmaya devam edin."
        assert classify_chunk(text) is ChunkCategory.ROSTER

    def test_submission_notes_are_submission(self):
        text = (
            "GitHub repo hazır ama video çekmedim → teslim eksik sayılır, video zorunlu. "
            "Linklerin e-posta ile gönderilmesi gerekir."
        )
        assert classify_chunk(text, source="sample_submission_notes") is ChunkCategory.SUBMISSION

    def test_project_name_in_roster_is_not_technical(self):
        text = (
            "Son isim listesini pinli mesajdaki adreste bulabilirsiniz. "
            "Projeler: Building Your First Local RAG Application with Foundry Local"
        )
        assert classify_chunk(text) is not ChunkCategory.TECHNICAL


class TestQueryTopic:
    def test_liste_devam_query_is_roster(self):
        topic = infer_query_topic(
            "Liste güncel değilse çalışmaya devam etmeli miyim?",
            AssistantMode.INSTRUCTOR_QA,
        )
        assert topic.value == "roster"

    def test_technical_mode_wins_even_if_query_mentions_liste(self):
        topic = infer_query_topic(
            "Foundry Local çalışmazsa ne kontrol etmeliyim?",
            AssistantMode.TECHNICAL_HELP,
        )
        assert topic.value == "technical"

    def test_checklist_mode_is_submission(self):
        topic = infer_query_topic(
            "GitHub repo hazır ama video çekmedim",
            AssistantMode.SUBMISSION_CHECKLIST,
        )
        assert topic.value == "submission"

    def test_github_contents_query_is_submission(self):
        topic = infer_query_topic(
            "GitHub'a ne koymam gerekiyor?",
            AssistantMode.INSTRUCTOR_QA,
        )
        assert topic.value == "submission"

    def test_whatsapp_channel_query_is_not_general(self):
        topic = infer_query_topic(
            "WhatsApp'tan yazmam yeterli mi?",
            AssistantMode.INSTRUCTOR_QA,
        )
        assert topic.value in {"submission", "correction"}


class TestFilterScoredChunks:
    def test_roster_query_drops_technical_chunks(self):
        roster = ScoredChunk(_chunk("Liste gecikmeli güncelleniyor, siz çalışmaya devam edin."), 0.4)
        technical = ScoredChunk(
            _chunk(
                "Foundry Local kurdum ama uygulama modele bağlanamıyor. "
                "Önce servisin ayakta olduğundan emin olun."
            ),
            0.39,
        )
        kept = filter_scored_chunks(
            [roster, technical],
            "Liste güncel değilse çalışmaya devam etmeli miyim?",
            AssistantMode.INSTRUCTOR_QA,
        )
        assert [s.chunk.category for s in kept] == [ChunkCategory.ROSTER]
        assert "bağlanamıyor" not in " ".join(s.chunk.text for s in kept)

    def test_technical_query_keeps_technical_chunks(self):
        technical = ScoredChunk(
            _chunk(
                "Uygulama modele bağlanamıyor. Önce servisin ayakta olduğundan emin olun."
            ),
            0.5,
        )
        roster = ScoredChunk(_chunk("Liste gecikmeli güncelleniyor, siz çalışmaya devam edin."), 0.2)
        kept = filter_scored_chunks(
            [technical, roster],
            "Foundry Local çalışmazsa ne kontrol etmeliyim?",
            AssistantMode.TECHNICAL_HELP,
        )
        assert any(s.chunk.category is ChunkCategory.TECHNICAL for s in kept)
        assert all(s.chunk.category is not ChunkCategory.ROSTER for s in kept)
