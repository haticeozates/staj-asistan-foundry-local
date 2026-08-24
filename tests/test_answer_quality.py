"""Answer-quality polish: category filter, extractive format, citation pruning."""

from __future__ import annotations

from staj_asistan.generation import ExtractiveClient
from staj_asistan.models import AssistantMode, ChunkCategory, EvidenceLevel
from staj_asistan.pipeline import Assistant

LISTE_QUERY = "Liste güncel değilse çalışmaya devam etmeli miyim?"
CHECKLIST_QUERY = "GitHub repo hazır ama video çekmedim, teslim için eksiğim var mı?"
FOUNDRY_QUERY = "Foundry Local çalışmazsa ne kontrol etmeliyim?"

TECHNICAL_LEAK_MARKERS = (
    "bağlanamıyor",
    "modele bağlan",
    "değerlendirirken ilk oraya bakıyorum",
    "download_and_register",
    "execution provider",
    "vektör veritabanına",
)


def _extractive_assistant() -> Assistant:
    instance = Assistant(llm=ExtractiveClient())
    instance.load_samples()
    return instance


def _blob(answer) -> str:
    parts = [answer.text]
    parts.extend(c.snippet for c in answer.citations)
    parts.extend(s.chunk.text for s in answer.retrieved)
    return " ".join(parts).lower()


class TestListeQueryStaysOnRoster:
    def test_liste_devam_query_does_not_return_technical_chunks(self):
        answer = _extractive_assistant().ask(LISTE_QUERY)
        blob = _blob(answer)
        for marker in TECHNICAL_LEAK_MARKERS:
            assert marker not in blob, f"technical leak: {marker!r}"
        assert all(
            s.chunk.category is not ChunkCategory.TECHNICAL for s in answer.retrieved
        )

    def test_liste_devam_short_answer_says_keep_working(self):
        answer = _extractive_assistant().ask(LISTE_QUERY)
        text = answer.text.lower()
        assert "kısa cevap:" in text
        short = text.split("ne yapmalısın:")[0]
        assert "devam" in short
        assert "edin" in short or "etmelisin" in short or "edilmeli" in short


class TestSubmissionAndTechnicalRouting:
    def test_final_teslim_canonical_rules_not_anecdote(self):
        answer = _extractive_assistant().ask("Final tesliminde ne gerekiyor?")
        text = answer.text.lower()
        short = text.split("ne yapmalısın:")[0] if "ne yapmalısın:" in text else text
        assert "github" in short or "kaynak kod" in short or "source" in short
        assert "video" in short
        assert "e-posta" in short or "e-mail" in short or "eposta" in short
        assert "çekmedim" not in short
        assert "repom hazır" not in short
        assert answer.mode is AssistantMode.SUBMISSION_CHECKLIST
        assert "kaynak kod" in text or "github" in text
        assert "whatsapp" in text or "e-posta" in text

    def test_final_teslim_prefers_submission_evidence(self):
        answer = _extractive_assistant().ask("Final tesliminde ne gerekiyor?")
        assert any(s.chunk.category is ChunkCategory.SUBMISSION for s in answer.retrieved)
        short = answer.text.lower().split("ne yapmalısın:")[0]
        assert "video" in short or "github" in short

    def test_missing_video_question_retrieves_submission_sources(self):
        answer = _extractive_assistant().ask(
            CHECKLIST_QUERY, mode=AssistantMode.SUBMISSION_CHECKLIST
        )
        sources = " ".join(s.chunk.source.lower() for s in answer.retrieved)
        categories = {s.chunk.category for s in answer.retrieved}
        assert "submission" in sources or ChunkCategory.SUBMISSION in categories
        combined = " ".join(s.chunk.text.lower() for s in answer.retrieved)
        assert "video" in combined
        assert "github" in combined or "teslim" in combined
        assert all(s.chunk.category is not ChunkCategory.TECHNICAL for s in answer.retrieved)

    def test_foundry_outage_question_retrieves_technical_sources(self):
        answer = _extractive_assistant().ask(
            FOUNDRY_QUERY, mode=AssistantMode.TECHNICAL_HELP
        )
        assert answer.retrieved
        assert any(s.chunk.category is ChunkCategory.TECHNICAL for s in answer.retrieved)
        combined = " ".join(s.chunk.text.lower() for s in answer.retrieved)
        assert "bağlanamıyor" in combined or "servisin ayakta" in combined or "endpoint" in combined
        short = answer.text.lower().split("ne yapmalısın:")[0]
        assert "disconnected bir ortamda" not in short
        assert "servis" in short or "port" in short or "endpoint" in short or "model" in short


class TestExtractiveFormatAndCitations:
    def test_extractive_fallback_uses_readable_headings(self):
        answer = _extractive_assistant().ask(LISTE_QUERY)
        text = answer.text
        for heading in ("Kısa cevap:", "Ne yapmalısın:", "Kaynaklara göre kanıt:", "Kaynaklar:"):
            assert heading in text

    def test_citation_panel_is_pruned_to_top_sources(self):
        answer = _extractive_assistant().ask(LISTE_QUERY)
        assert 1 <= len(answer.citations) <= 3
        if answer.evidence is EvidenceLevel.HIGH:
            assert len(answer.citations) <= 2

    def test_extractive_quotes_only_come_from_retrieved_sources(self):
        assistant = _extractive_assistant()
        answer = assistant.ask(LISTE_QUERY)
        indexed = " ".join(s.chunk.text for s in answer.retrieved)
        evidence_block = answer.text.split("Kaynaklara göre kanıt:", 1)[-1].split("Kaynaklar:", 1)[0]
        for line in evidence_block.splitlines():
            line = line.lstrip("- ").rsplit(" [", 1)[0].strip()
            if len(line) < 20:
                continue
            assert line in indexed
