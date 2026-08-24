"""End-to-end pipeline tests, including the four demo scenarios.

These run fully offline: the hashing embedding backend and a stub LLM are used so
the suite is deterministic and needs neither a network nor a running model.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from staj_asistan.generation import ExtractiveClient, LLMClient, LLMUnavailable
from staj_asistan.ingestion import ingest_bytes, ingest_text, sample_directory
from staj_asistan.models import AssistantMode, EvidenceLevel
from staj_asistan.pipeline import Assistant

SAMPLES = sample_directory()


class StubLLM(LLMClient):
    """Echoes the prompt so tests can assert on what the model was given."""

    name = "stub"

    def __init__(self, reply: str = "Kaynaklara göre kısa cevap [1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.reply


class BrokenLLM(LLMClient):
    name = "broken"

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
        raise LLMUnavailable("model kapandı")


@pytest.fixture
def stub() -> StubLLM:
    return StubLLM()


@pytest.fixture
def assistant(stub) -> Assistant:
    instance = Assistant(llm=stub)
    instance.load_samples()
    return instance


class TestIndexing:
    def test_samples_are_indexed(self, assistant):
        stats = assistant.stats
        assert stats.file_count == 3
        assert stats.chunk_count > 0
        assert not assistant.is_empty

    def test_instructor_messages_are_counted(self, assistant):
        assert assistant.stats.instructor_message_count > 20

    def test_backend_is_reported(self, assistant):
        assert "hashing" in assistant.stats.embedding_backend

    def test_reset_clears_the_index(self, assistant):
        assistant.reset()
        assert assistant.is_empty
        assert assistant.stats.chunk_count == 0

    def test_zip_upload_reads_only_text_entries(self, tmp_path):
        import zipfile

        archive = tmp_path / "export.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("_chat.txt", "[1.07.2026 10:00:00] Eğitmen Microsoft: duyuru metni")
            zf.writestr("PHOTO-2026.jpg", b"\xff\xd8\xff\xe0binary")
        sources = ingest_bytes(archive.read_bytes(), "export.zip")
        assert len(sources) == 1
        assert sources[0].kind == "whatsapp"

    def test_plain_document_is_detected_as_document(self):
        source = ingest_text("Sadece düz metin bir not dosyası.", "notlar")
        assert source.kind == "document"


class TestGroundingContract:
    @pytest.mark.parametrize(
        "question",
        [
            "İstanbul'da hava nasıl?",
            "Real Madrid maçı kaçta?",
            "Türkiye'nin başkenti neresi?",
            "Kaç gün tatil yapacağız yazın?",
        ],
    )
    def test_out_of_scope_question_is_refused_without_calling_the_model(
        self, assistant, stub, question
    ):
        answer = assistant.ask(question)
        assert answer.evidence is EvidenceLevel.NONE
        assert answer.citations == []
        assert stub.calls == []
        assert "kaynaklarda net değil" in answer.text.lower()

    def test_empty_index_refuses_too(self, stub):
        empty = Assistant(llm=stub)
        answer = empty.ask("Sertifika süreci nasıl işliyor?")
        assert answer.evidence is EvidenceLevel.NONE
        assert stub.calls == []

    def test_in_scope_question_does_call_the_model(self, assistant, stub):
        assistant.ask("Final tesliminde ne gerekiyor?")
        assert len(stub.calls) == 1

    def test_prompt_contains_the_retrieved_sources(self, assistant, stub):
        assistant.ask("Sertifika süreci nasıl işliyor?")
        _, user_prompt = stub.calls[0]
        assert "KAYNAKLAR" in user_prompt
        assert "[1]" in user_prompt

    def test_prompt_forbids_inventing_information(self, assistant, stub):
        assistant.ask("Final tesliminde ne gerekiyor?")
        system_prompt, _ = stub.calls[0]
        assert "SADECE" in system_prompt

    def test_citation_markers_beyond_the_source_count_are_removed(self, assistant):
        assistant.llm = StubLLM("Cevap [1] ve uydurma kaynak [99].")
        answer = assistant.ask("Final tesliminde ne gerekiyor?")
        assert "[99]" not in answer.text
        assert "[1]" in answer.text

    def test_answer_reports_whether_it_cited_anything(self, assistant):
        assistant.llm = StubLLM("Kaynak göstermeyen bir cevap.")
        assert assistant.ask("Final tesliminde ne gerekiyor?").grounded is False

    def test_model_failure_degrades_to_extractive_instead_of_erroring(self, assistant):
        assistant.llm = BrokenLLM()
        answer = assistant.ask("Final tesliminde ne gerekiyor?")
        assert answer.citations
        assert "Alıntı modu" in answer.generator

    def test_extractive_client_only_quotes_existing_sentences(self):
        instance = Assistant(llm=ExtractiveClient())
        instance.load_samples()
        answer = instance.ask("Final tesliminde ne gerekiyor?")
        indexed = " ".join(c.text for c in instance.store.all_chunks())
        quoted = [
            line.lstrip("- ").rsplit(" [", 1)[0]
            for line in answer.text.splitlines()
            if line.startswith("- ") and re.search(r"\[\d+\]\s*$", line)
        ]
        assert quoted
        for sentence in quoted:
            if "kaynaklarda doğrudan" in sentence.lower():
                continue
            assert sentence in indexed


class TestCitations:
    def test_citations_are_numbered_from_one(self, assistant):
        answer = assistant.ask("Sertifika süreci nasıl işliyor?")
        assert [c.index for c in answer.citations] == list(range(1, len(answer.citations) + 1))

    def test_citations_carry_source_and_snippet(self, assistant):
        citation = assistant.ask("Final tesliminde ne gerekiyor?").citations[0]
        assert citation.source
        assert citation.snippet
        assert citation.label.count("·") == 2

    def test_snippets_are_truncated(self, assistant):
        for citation in assistant.ask("Final tesliminde ne gerekiyor?").citations:
            assert len(citation.snippet) <= 340

    def test_no_personal_data_reaches_the_citations(self, assistant):
        import re

        for question in ("Final tesliminde ne gerekiyor?", "Sertifika ne zaman gelir?"):
            for citation in assistant.ask(question).citations:
                assert not re.search(r"[\w.+-]+@[\w-]+\.\w+", citation.snippet)


class TestDemoScenarios:
    """The four questions used in the final presentation."""

    def test_demo_1_final_submission_requirements(self, assistant):
        answer = assistant.ask("Final tesliminde ne gerekiyor?")
        assert answer.evidence in {EvidenceLevel.HIGH, EvidenceLevel.MEDIUM}
        combined = " ".join(c.snippet.lower() for c in answer.citations)
        assert "video" in combined
        assert "github" in combined or "link" in combined

    def test_demo_2_should_i_keep_working_if_the_list_is_stale(self, assistant):
        answer = assistant.ask("Liste güncel değilse çalışmaya devam etmeli miyim?")
        assert answer.citations
        combined = " ".join(c.snippet.lower() for c in answer.citations)
        assert "devam" in combined or "gecikmeli" in combined

    def test_demo_3_checklist_with_missing_video(self, assistant):
        answer = assistant.ask(
            "GitHub repo hazır ama video çekmedim, teslim için eksiğim var mı?",
            mode=AssistantMode.SUBMISSION_CHECKLIST,
        )
        assert answer.structured["teslime_hazir"] is False
        assert any("video" in item.lower() for item in answer.structured["eksikler"])
        assert any("Kaynak kod" in item for item in answer.structured["hazir_olanlar"])

    def test_demo_4_correction_request_is_not_enough(self, assistant):
        answer = assistant.ask(
            "Listede projem yanlış görünüyor, Foundry Local olarak güncellenmesini istiyorum. "
            "Bu mesaj yeterli mi?",
            mode=AssistantMode.CORRECTION_ANALYZER,
        )
        assert answer.structured["intent"] == "list_correction"
        assert "full_name" in answer.structured["missing_required_info"]
        assert answer.structured["auto_apply"] is False
        assert "json" in answer.text.lower()


class TestModes:
    def test_checklist_mode_returns_structured_output(self, assistant):
        answer = assistant.ask("Her şey hazır", mode=AssistantMode.SUBMISSION_CHECKLIST)
        assert answer.mode is AssistantMode.SUBMISSION_CHECKLIST
        assert "teslime_hazir" in answer.structured

    def test_checklist_works_even_without_matching_sources(self, stub):
        instance = Assistant(llm=stub)
        answer = instance.ask("video çekmedim", mode=AssistantMode.SUBMISSION_CHECKLIST)
        assert answer.structured["teslime_hazir"] is False
        assert "varsayılan kurallara" in answer.text

    def test_correction_mode_includes_an_email_draft(self, assistant):
        answer = assistant.ask(
            "Listede adım yanlış, projem Foundry Local olmalı",
            mode=AssistantMode.CORRECTION_ANALYZER,
        )
        assert "E-posta taslağı" in answer.text
        assert "gönderilmedi" in answer.text

    def test_technical_mode_uses_a_technical_system_prompt(self, assistant, stub):
        assistant.ask("Foundry Local GPU yerine CPU kullanıyor", mode=AssistantMode.TECHNICAL_HELP)
        assert "teknik rehberlik" in stub.calls[0][0]

    def test_each_mode_has_its_own_system_prompt(self, assistant, stub):
        for mode in AssistantMode:
            assistant.ask("Foundry Local ve teslim hakkında bilgi", mode=mode)
        prompts = {call[0] for call in stub.calls}
        assert len(prompts) >= 3


class TestEvidenceLevels:
    def test_strong_match_yields_high_or_medium(self, assistant):
        answer = assistant.ask("Sertifika tek mi, iki proje yaparsam iki sertifika alır mıyım?")
        assert answer.evidence in {EvidenceLevel.HIGH, EvidenceLevel.MEDIUM}

    def test_no_match_yields_none(self, assistant):
        assert assistant.ask("Real Madrid maçı kaçta?").evidence is EvidenceLevel.NONE

    def test_blank_question_is_handled(self, assistant):
        answer = assistant.ask("   ")
        assert answer.evidence is EvidenceLevel.NONE
        assert "soru" in answer.text.lower()


RUN_MODEL_TESTS = os.getenv("STAJ_ASISTAN_RUN_MODEL_TESTS", "").strip() in {"1", "true", "yes"}

OFF_TOPIC_QUESTIONS = [
    "Mars'a ne zaman insan gönderilecek?",
    "Bugün borsa ne olacak?",
    "İstanbul'da hava nasıl?",
    "En iyi pizza tarifi nedir?",
    "Kedimin aşı takvimi ne olmalı?",
    "Real Madrid maçı kaçta?",
    "Vize başvurusu için ne zaman randevu alınır?",
    "Türkiye'nin başkenti neresi?",
]

ON_TOPIC_QUESTIONS = [
    "Final tesliminde ne gerekiyor?",
    "Liste güncel değilse çalışmaya devam etmeli miyim?",
    "Sertifika süreci nasıl işliyor?",
    "GitHub'a ne koymam gerekiyor?",
    "Foundry Local bu yıl neden öneriliyor?",
    "Proje seçenekleri neler?",
    "Video zorunlu mu?",
    "Python ile geliştirebilir miyim?",
    "Foundry Local çalışmazsa ne kontrol etmeliyim?",
    "Local RAG mimarisi nasıl kurulur?",
]


@pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="Loads a real embedding model; enable with STAJ_ASISTAN_RUN_MODEL_TESTS=1",
)
class TestSemanticBackendSeparation:
    """Verifies the *primary* retrieval path, not just the offline fallback.

    The hashing fallback cannot tell "Mars'a ne zaman insan gönderilecek?" apart from
    a program question, because it shares the stems "zaman" and "gönder". A real
    sentence-embedding backend can, and this is where that claim is checked.
    """

    @pytest.fixture(scope="class")
    def semantic_assistant(self):
        from staj_asistan.embeddings import SentenceTransformerEmbeddings

        instance = Assistant(embedding_backend=SentenceTransformerEmbeddings(), llm=StubLLM())
        instance.load_samples()
        return instance

    @pytest.mark.parametrize("question", OFF_TOPIC_QUESTIONS)
    def test_off_topic_questions_are_refused(self, semantic_assistant, question):
        assert semantic_assistant.ask(question).evidence is EvidenceLevel.NONE

    @pytest.mark.parametrize("question", ON_TOPIC_QUESTIONS)
    def test_on_topic_questions_are_answered(self, semantic_assistant, question):
        assert semantic_assistant.ask(question).citations


class TestSampleDataHygiene:
    """Sample data is committed to GitHub, so it must be provably clean."""

    def test_sample_directory_exists(self):
        assert SAMPLES.is_dir()
        assert list(SAMPLES.glob("*.txt"))

    @pytest.mark.parametrize("path", sorted(SAMPLES.glob("*.txt")) if SAMPLES.is_dir() else [])
    def test_samples_contain_no_real_contact_details(self, path: Path):
        import re

        text = path.read_text(encoding="utf-8")
        emails = [e.rstrip(".") for e in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)]
        assert all(e.endswith("example.com") for e in emails), emails
        assert not re.search(r"\+\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}", text)
        # Any link at all is suspicious in sample data unless it is public documentation.
        links = re.findall(r"https?://[^\s)]+", text)
        allowed = ("learn.microsoft.com", "github.com", "example.com", "example.net")
        assert all(any(host in link for host in allowed) for link in links), links
