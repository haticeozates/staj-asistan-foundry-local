"""Workflow tests: the deterministic checklist and correction analyzer."""

from __future__ import annotations

import pytest

from staj_asistan.models import AssistantMode
from staj_asistan.workflows import (
    analyze_correction_request,
    analyze_submission,
    detect_state,
    suggest_mode,
)


class TestStateDetection:
    def test_positive_mention(self):
        assert detect_state("GitHub repo hazır", ("github",)) is True

    def test_negated_mention(self):
        assert detect_state("video çekmedim", ("video",)) is False

    def test_absent_mention(self):
        assert detect_state("GitHub repo hazır", ("video",)) is None

    def test_negation_wins_over_positive_in_another_clause(self):
        assert detect_state("video düşündüm ama video çekmedim", ("video",)) is False

    @pytest.mark.parametrize(
        "text", ["video yok", "video hazır değil", "videoyu henüz çekmedim", "video eksik"]
    )
    def test_negation_variants(self, text):
        assert detect_state(text, ("video",)) is False


class TestSubmissionChecklist:
    STATUS = "GitHub repo hazır, README var ama video çekmedim."

    def test_ready_items_are_detected(self):
        result = analyze_submission(self.STATUS)
        assert any("Kaynak kod" in item for item in result.ready)
        assert any("README" in item for item in result.ready)

    def test_missing_video_is_detected(self):
        result = analyze_submission(self.STATUS)
        assert any("demo videosu" in item for item in result.missing)

    def test_unmentioned_required_item_is_reported_as_missing(self):
        result = analyze_submission(self.STATUS)
        assert any("e-posta" in item.lower() for item in result.missing)

    def test_next_steps_are_actionable_and_ordered(self):
        result = analyze_submission(self.STATUS)
        assert result.next_steps
        assert any("video" in step.lower() for step in result.next_steps)

    def test_incomplete_submission_is_not_marked_complete(self):
        assert analyze_submission(self.STATUS).complete is False

    def test_complete_submission_is_recognised(self):
        result = analyze_submission(
            "GitHub repom hazır, 2 dakikalık videoyu çektim ve linkleri mail attım."
        )
        assert result.complete is True
        assert not [item for item in result.missing if "belirtilmemiş" in item]

    def test_structured_output_shape(self):
        data = analyze_submission(self.STATUS).as_dict()
        assert set(data) == {
            "hazir_olanlar",
            "eksikler",
            "belirsizler",
            "onerilen_adimlar",
            "teslime_hazir",
        }

    def test_text_rendering_has_all_sections(self):
        text = analyze_submission(self.STATUS).as_text()
        assert "**Hazır olanlar**" in text
        assert "**Eksikler**" in text
        assert "**Önerilen sonraki adımlar**" in text

    def test_empty_status_reports_everything_missing(self):
        result = analyze_submission("")
        assert len(result.missing) >= 3
        assert result.complete is False


class TestCorrectionAnalyzer:
    MESSAGE = "Listede adım yanlış, projem Foundry Local olmalı, devam etmek istiyorum."

    def test_intent_is_list_correction(self):
        assert analyze_correction_request(self.MESSAGE).intent == "list_correction"

    def test_project_is_mapped_to_the_catalog_name(self):
        fields = analyze_correction_request(self.MESSAGE).detected_fields
        assert fields["project"] == "Building Your First Local RAG Application with Foundry Local"

    def test_status_is_detected(self):
        fields = analyze_correction_request(self.MESSAGE).detected_fields
        assert fields["status"] == "Devam etmek istiyorum"

    def test_negative_status_is_not_confused_with_positive(self):
        fields = analyze_correction_request("Devam etmek istemiyorum").detected_fields
        assert fields["status"] == "Devam etmek istemiyorum"

    def test_missing_identity_is_reported(self):
        analysis = analyze_correction_request(self.MESSAGE)
        assert analysis.missing_required_info == ["full_name", "email"]

    def test_recommendation_points_to_email(self):
        assert "e-posta" in analyze_correction_request(self.MESSAGE).recommendation.lower()

    def test_structured_output_matches_the_agreed_schema(self):
        data = analyze_correction_request(self.MESSAGE).as_dict()
        assert set(data) == {
            "intent",
            "detected_fields",
            "missing_required_info",
            "recommendation",
            "confidence",
            "auto_apply",
        }

    def test_analyzer_never_applies_the_change(self):
        assert analyze_correction_request(self.MESSAGE).as_dict()["auto_apply"] is False

    def test_complete_request_has_no_missing_fields(self):
        analysis = analyze_correction_request(
            "Adım Örnek Katılımcı, e-posta ornek@example.com. "
            "Listede projem Quantum Kickstart görünüyor, PyTorch olarak güncelleyin."
        )
        assert analysis.missing_required_info == []
        assert analysis.confidence == "high"

    def test_email_value_is_never_stored(self):
        analysis = analyze_correction_request("ornek@example.com adresiyle kayıtlıyım, listede yokum")
        assert analysis.detected_fields["email"] == "mesajda belirtilmiş"
        assert "ornek@example.com" not in str(analysis.as_dict())

    def test_name_is_extracted_from_common_phrasings(self):
        for phrasing in ("Adım Örnek Katılımcı", "İsmim Örnek Katılımcı", "Ben Örnek Katılımcı"):
            fields = analyze_correction_request(f"{phrasing}, listede yokum").detected_fields
            assert fields.get("full_name") == "Örnek Katılımcı"

    def test_language_change_is_detected(self):
        fields = analyze_correction_request("Dilimi English olarak güncelleyin").detected_fields
        assert fields["language"] == "English"

    def test_intent_falls_back_when_nothing_matches(self):
        assert analyze_correction_request("merhaba").intent == "unknown"

    def test_draft_email_uses_placeholders_when_identity_is_missing(self):
        draft = analyze_correction_request(self.MESSAGE).suggested_email
        assert "<Ad Soyad>" in draft
        assert "Building Your First Local RAG Application with Foundry Local" in draft

    def test_draft_email_uses_the_detected_name(self):
        analysis = analyze_correction_request("Adım Örnek Katılımcı, listede projem yanlış")
        assert "Örnek Katılımcı" in analysis.suggested_email

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Foundry Local RAG projesine geçmek istiyorum", "Building Your First Local RAG Application with Foundry Local"),
            ("PyTorch ile hisse fiyat tahmini yapacağım", "Stock Price Prediction with PyTorch"),
            ("Q# kuantum projesini seçtim", "Quantum Kickstart with Q#"),
            ("Titanic verisiyle çalışıyorum", "Titanic Survival Analysis"),
            ("Kendi projem olacak", "Kendi projem"),
        ],
    )
    def test_full_project_catalog_is_recognised(self, text, expected):
        assert analyze_correction_request(text).detected_fields["project"] == expected


class TestModeSuggestion:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Listede projem yanlış görünüyor", AssistantMode.CORRECTION_ANALYZER),
            ("GitHub repo hazır ama video çekmedim", AssistantMode.SUBMISSION_CHECKLIST),
            ("Foundry Local endpoint hatası alıyorum", AssistantMode.TECHNICAL_HELP),
            ("Sertifika ne zaman gelir", AssistantMode.INSTRUCTOR_QA),
        ],
    )
    def test_mode_hints(self, text, expected):
        assert suggest_mode(text) is expected
