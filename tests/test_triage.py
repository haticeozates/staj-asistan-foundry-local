"""Incoming-message triage: intent classification, reply policy, and the no-send guarantees.

These tests encode the product rule that makes the assistant safe to point at a real
group: it may draft, it may refuse, it may escalate — it may never send.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from staj_asistan.generation import ExtractiveClient
from staj_asistan.models import AssistantMode, ChunkCategory, EvidenceLevel
from staj_asistan.pipeline import Assistant
from staj_asistan.triage import (
    Channel,
    IncomingMessage,
    MessageIntent,
    ReplyDecision,
    TriageResult,
    classify_intent,
    triage,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "staj_asistan"

SUBMISSION_STATUS = "GitHub repo hazır ama video çekmedim, teslim olur mu?"
CORRECTION = "Listede projem yanlış görünüyor, Foundry Local olarak güncellenmesini istiyorum."
CHANNEL_QUESTION = "WhatsApp'tan yazmam yeterli mi?"
CERTIFICATE = "Sertifikam ne zaman gelir?"
OFF_TOPIC = "İstanbul'da hava nasıl?"


@pytest.fixture(scope="module")
def assistant() -> Assistant:
    instance = Assistant(llm=ExtractiveClient())
    instance.load_samples()
    return instance


def _message(text: str, channel: Channel = Channel.WHATSAPP) -> IncomingMessage:
    return IncomingMessage(
        text=text,
        channel=channel,
        group="AI Innovators - Örnek Grup",
        sender_alias="Katılımcı#0001",
    )


class TestIntentClassification:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Final tesliminde ne gerekiyor?", MessageIntent.SUBMISSION_REQUIREMENT),
            (SUBMISSION_STATUS, MessageIntent.SUBMISSION_REQUIREMENT),
            (CHANNEL_QUESTION, MessageIntent.SUBMISSION_CHANNEL),
            (CORRECTION, MessageIntent.CORRECTION_REQUEST),
            (
                "Liste güncel değilse çalışmaya devam etmeli miyim?",
                MessageIntent.ROSTER_CONTINUE,
            ),
            (
                "Foundry Local çalışmazsa ne kontrol etmeliyim?",
                MessageIntent.TECHNICAL_HELP,
            ),
            (CERTIFICATE, MessageIntent.CERTIFICATE),
            (OFF_TOPIC, MessageIntent.UNKNOWN),
        ],
    )
    def test_intents(self, text, expected):
        assert classify_intent(text) is expected

    def test_blank_message_is_unknown(self):
        assert classify_intent("   ") is MessageIntent.UNKNOWN

    def test_intent_selects_the_matching_mode(self, assistant):
        result = triage(_message(CORRECTION), assistant)
        assert result.selected_mode is AssistantMode.CORRECTION_ANALYZER


class TestReplyPolicy:
    def test_submission_question_is_drafted(self, assistant):
        result = triage(_message(SUBMISSION_STATUS), assistant)
        assert result.reply_decision is ReplyDecision.DRAFT_REPLY
        assert result.draft_reply.strip()
        assert result.evidence_level in {EvidenceLevel.HIGH, EvidenceLevel.MEDIUM}

    def test_correction_request_always_needs_a_human(self, assistant):
        result = triage(_message(CORRECTION), assistant)
        assert result.reply_decision is ReplyDecision.NEEDS_HUMAN_APPROVAL
        assert result.requires_human is True

    def test_certificate_question_needs_a_human(self, assistant):
        result = triage(_message(CERTIFICATE), assistant)
        assert result.reply_decision is ReplyDecision.NEEDS_HUMAN_APPROVAL
        assert "sertifika" in result.reason.lower()

    def test_deadline_question_needs_a_human(self, assistant):
        result = triage(_message("Teslim son tarihini benim için uzatabilir misiniz?"), assistant)
        assert result.reply_decision is ReplyDecision.NEEDS_HUMAN_APPROVAL

    def test_message_carrying_personal_data_needs_a_human(self, assistant):
        result = triage(
            _message("Final tesliminde ne gerekiyor? Bana ornek@example.com adresinden dönün."),
            assistant,
        )
        assert result.reply_decision is ReplyDecision.NEEDS_HUMAN_APPROVAL
        assert "kişisel" in result.reason.lower()

    def test_no_evidence_is_never_answered(self, assistant):
        result = triage(_message(OFF_TOPIC), assistant)
        assert result.reply_decision is ReplyDecision.DO_NOT_ANSWER
        assert result.evidence_level is EvidenceLevel.NONE

    def test_do_not_answer_produces_no_draft_at_all(self, assistant):
        result = triage(_message(OFF_TOPIC), assistant)
        assert result.draft_reply == ""
        assert "eğitmen" in result.reason.lower()

    def test_rule_backed_answer_with_weak_sources_is_escalated_not_dropped(self, assistant):
        # The channel card is deterministic, so low retrieval evidence means "thin
        # citations", not "unreliable answer". Escalate to a human instead of refusing.
        result = triage(_message(CHANNEL_QUESTION), assistant)
        assert result.intent is MessageIntent.SUBMISSION_CHANNEL
        assert result.reply_decision is ReplyDecision.NEEDS_HUMAN_APPROVAL
        assert result.draft_reply.strip()

    def test_every_result_reports_source_categories(self, assistant):
        result = triage(_message(SUBMISSION_STATUS), assistant)
        assert result.source_categories
        assert all(isinstance(c, ChunkCategory) for c in result.source_categories)

    def test_result_is_serialisable_for_the_ui(self, assistant):
        data = triage(_message(SUBMISSION_STATUS), assistant).as_dict()
        assert set(data) >= {
            "intent",
            "selected_mode",
            "evidence_level",
            "source_categories",
            "reply_decision",
            "reason",
            "auto_send",
        }
        assert data["auto_send"] is False


class TestChannelIsMetadataOnly:
    @pytest.mark.parametrize("channel", list(Channel))
    def test_decision_does_not_depend_on_the_channel(self, assistant, channel):
        baseline = triage(_message(SUBMISSION_STATUS, Channel.MANUAL), assistant)
        result = triage(_message(SUBMISSION_STATUS, channel), assistant)
        assert result.reply_decision is baseline.reply_decision
        assert result.draft_reply == baseline.draft_reply

    def test_channel_values_are_the_agreed_three(self):
        assert {c.value for c in Channel} == {"whatsapp", "telegram", "manual"}

    def test_triage_reads_no_environment_or_token(self):
        source = (SRC / "triage.py").read_text(encoding="utf-8")
        for forbidden in ("os.getenv", "os.environ", "TOKEN", "api_key", "Bearer"):
            assert forbidden not in source, f"triage.py must not reference {forbidden!r}"


class TestNoAutoSendGuarantee:
    def test_no_outbound_decision_exists(self):
        values = {d.value for d in ReplyDecision}
        assert values == {"draft_reply", "needs_human_approval", "do_not_answer"}

    def test_triage_module_exposes_no_sending_callable(self):
        import staj_asistan.triage as module

        for name in dir(module):
            assert not re.match(r"^(send|post|publish|deliver|reply_to)", name)

    def test_no_module_targets_a_messaging_api(self):
        forbidden = (
            "api.telegram.org",
            "graph.facebook.com",
            "sendMessage",
            "TELEGRAM_BOT_TOKEN",
            "WHATSAPP_TOKEN",
            "WHATSAPP_ACCESS_TOKEN",
        )
        files = list(SRC.glob("*.py")) + [ROOT / "app.py"]
        for path in files:
            source = path.read_text(encoding="utf-8")
            for needle in forbidden:
                assert needle not in source, f"{path.name} references {needle!r}"

    def test_ui_has_no_send_button(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        assert not re.search(r"button\(\s*[\"'][^\"']*(?:[Gg]önder|[Ss]end)", source)

    def test_triage_never_touches_the_private_corpus(self):
        for path in ((SRC / "triage.py"), (ROOT / "app.py")):
            assert "data/private" not in path.read_text(encoding="utf-8")

    def test_result_type_has_no_send_method(self):
        assert not [name for name in dir(TriageResult) if name.startswith("send")]


class TestSimulatorView:
    """Renders the Streamlit view headlessly. The simulator is the demo; a crash here
    is the kind that is only discovered in front of an audience."""

    @pytest.fixture(scope="class")
    def app(self):
        pytest.importorskip("streamlit", reason="UI extra not installed")
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        next(b for b in at.sidebar.button if b.label == "Örnek veri").click().run()
        at.sidebar.radio[0].set_value("Gelen Mesaj Simülasyonu").run()
        return at

    def _analyse(self, app, text: str):
        app.text_area(key="incoming_text").set_value(text)
        return next(b for b in app.button if b.label == "Mesajı analiz et").click().run()

    def test_view_warns_that_nothing_is_ever_sent(self, app):
        assert any("otomatik mesaj göndermez" in w.value for w in app.warning)

    def test_drafted_answer_is_shown_as_copyable_code(self, app):
        rendered = self._analyse(app, SUBMISSION_STATUS)
        assert not rendered.exception
        assert len(rendered.code) == 1
        assert any("Taslak hazır" in s.value for s in rendered.success)

    def test_refused_message_renders_no_draft_block(self, app):
        rendered = self._analyse(app, OFF_TOPIC)
        assert not rendered.exception
        assert rendered.code == []
