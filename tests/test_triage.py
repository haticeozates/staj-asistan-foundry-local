"""Incoming-message triage: intent classification, reply policy, and the no-send guarantees.

These tests encode the product rule that makes the assistant safe to point at a real
group: it may draft, it may refuse, it may escalate — it may never send.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from staj_asistan.approval_queue import ApprovalQueue, QueueStatus
from staj_asistan.corpus import build_private_index
from staj_asistan.embeddings import HashingEmbeddings
from staj_asistan.generation import ExtractiveClient
from staj_asistan.models import AssistantMode, ChunkCategory, EvidenceLevel
from staj_asistan.pipeline import Assistant
from staj_asistan.private_config import PrivateConfig
from staj_asistan.telegram import TelegramDeliveryRejected, TelegramDeliveryUncertain
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

PRIVATE_CONFIG_ENV = "STAJ_ASISTAN_PRIVATE_CONFIG"
SYNTHETIC_CHAT_ID = -1009876543210
SYNTHETIC_ALIAS = "Örnek Eğitmen"
SYNTHETIC_EXPORT = (
    "[30.09.2026 10:00:00] Örnek Eğitmen: Teslim için repo bağlantısı ve kısa video gereklidir.\n"
    "[30.09.2026 10:01:00] Örnek Katılımcı: Teşekkürler.\n"
)
SEND_LABEL = "Onayla ve Telegram'a gönder"
REJECT_LABEL = "Reddet"


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

    def test_correction_message_matches_the_assistant_view_verdict(self, assistant):
        # Same message, two entry points: the simulator escalates it and the plain
        # assistant view answers it through the same correction workflow.
        result = triage(_message(CORRECTION), assistant)
        assert result.reply_decision is ReplyDecision.NEEDS_HUMAN_APPROVAL
        assert result.selected_mode is assistant.ask(CORRECTION).mode

    def test_off_topic_message_is_refused_with_nothing_attached(self, assistant):
        result = triage(_message(OFF_TOPIC), assistant)
        assert result.reply_decision is ReplyDecision.DO_NOT_ANSWER
        assert result.draft_reply == ""
        assert result.source_categories == ()
        assert result.answer.citations == []

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
        telegram_needles = ("api.telegram.org", "sendMessage", "TELEGRAM_BOT_TOKEN")
        forbidden = telegram_needles + (
            "graph.facebook.com",
            "WHATSAPP_TOKEN",
            "WHATSAPP_ACCESS_TOKEN",
        )
        # telegram.py is the single, approval-gated Telegram boundary.
        files = list(SRC.glob("*.py")) + [ROOT / "app.py"]
        for path in files:
            source = path.read_text(encoding="utf-8")
            allowed = telegram_needles if path.name == "telegram.py" else ()
            for needle in forbidden:
                if needle in allowed:
                    continue
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

        with pytest.MonkeyPatch.context() as patch:
            patch.delenv(PRIVATE_CONFIG_ENV, raising=False)
            patch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
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

    def test_assistant_view_announces_the_workflow_it_routed_to(self, app):
        app.sidebar.radio[0].set_value("Asistan").run()
        app.text_area(key="question_text").set_value(CORRECTION)
        rendered = next(b for b in app.button if b.label == "Sor").click().run()
        assert not rendered.exception
        assert any("Düzeltme İsteği Analizi** kapsamında" in i.value for i in rendered.info)


def _queue_page(queue, transport_factory, intent_labels):
    from staj_asistan.approval_ui import render_approval_view

    render_approval_view(queue, transport_factory, intent_labels=intent_labels)


class _FakeTransport:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.sent: list[tuple[int, str, int]] = []

    def get_updates(self, offset, timeout):  # pragma: no cover - must never be called
        raise AssertionError("the approval view must never poll Telegram")

    def send_message(self, chat_id: int, text: str, reply_to_message_id: int) -> int:
        self.sent.append((chat_id, text, reply_to_message_id))
        if self.error is not None:
            raise self.error
        return 4242


class _TransportFactory:
    def __init__(self, transport: _FakeTransport):
        self.transport = transport
        self.calls = 0

    def __call__(self) -> _FakeTransport:
        self.calls += 1
        return self.transport


def _rendered_text(at) -> str:
    parts: list[str] = []
    for kind in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "text",
        "code",
        "info",
        "warning",
        "error",
        "success",
        "text_area",
    ):
        parts.extend(str(element.value) for element in getattr(at, kind))
    parts.extend(str(element.label) for element in at.button)
    parts.extend(str(element.label) for element in at.expander)
    parts.extend(f"{metric.label} {metric.value}" for metric in at.metric)
    return "\n".join(parts)


class TestApprovalQueueView:
    """The approval queue is the one screen that can reach a real group, so every
    control it renders is part of the send-safety contract."""

    @pytest.fixture(autouse=True)
    def no_network(self, monkeypatch):
        pytest.importorskip("streamlit", reason="UI extra not installed")
        import requests

        def refuse(*_args, **_kwargs):
            raise AssertionError("tests must not open network connections")

        monkeypatch.setattr(requests.Session, "request", refuse)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv(PRIVATE_CONFIG_ENV, raising=False)

    @pytest.fixture
    def queue(self, tmp_path) -> ApprovalQueue:
        return ApprovalQueue(tmp_path / "queue.sqlite3")

    @pytest.fixture
    def draft_result(self, assistant):
        result = triage(_message(SUBMISSION_STATUS, Channel.TELEGRAM), assistant)
        assert result.reply_decision is ReplyDecision.DRAFT_REPLY
        return result

    @pytest.fixture
    def refusal_result(self, assistant):
        result = triage(_message(OFF_TOPIC, Channel.TELEGRAM), assistant)
        assert result.reply_decision is ReplyDecision.DO_NOT_ANSWER
        return result

    @staticmethod
    def _enqueue(queue, result, update_id: int, text: str = SUBMISSION_STATUS):
        return queue.enqueue(
            update_id=update_id,
            chat_id=SYNTHETIC_CHAT_ID,
            message_id=5000 + update_id,
            incoming_text=text,
            sender_alias="Katılımcı#ab12",
            group_label="Telegram grubu#cd34",
            triage_result=result,
        )

    @staticmethod
    def _page(queue, factory):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_function(
            _queue_page,
            args=(queue, factory, {MessageIntent.SUBMISSION_REQUIREMENT: "Teslim kuralı / durumu"}),
            default_timeout=30,
        ).run()

    @staticmethod
    def _labels(at) -> list[str]:
        return [button.label for button in at.button]

    @staticmethod
    def _open_app():
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        return at.sidebar.radio[0].set_value("Onay Kuyruğu").run()

    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return self._open_app()

    @pytest.fixture
    def private_workspace(self, tmp_path, monkeypatch, refusal_result):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "instructor_aliases": [SYNTHETIC_ALIAS],
                    "allowed_telegram_chat_ids": [SYNTHETIC_CHAT_ID],
                    "source_labels": {"synthetic-chat.txt": "Sentetik Grup"},
                    "index_dir": "data/index/private",
                    "queue_db": "data/index/telegram-queue.sqlite3",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        export = tmp_path / "synthetic-chat.txt"
        export.write_text(SYNTHETIC_EXPORT, encoding="utf-8")
        build_private_index(
            PrivateConfig.load(config_path),
            [export],
            Path("data/index/private"),
            HashingEmbeddings(),
            allow_hashing=True,
        )
        monkeypatch.setenv(PRIVATE_CONFIG_ENV, str(config_path))
        return ApprovalQueue(tmp_path / "data" / "index" / "telegram-queue.sqlite3")

    @pytest.fixture
    def app_with_refusal(self, private_workspace, refusal_result):
        self._enqueue(private_workspace, refusal_result, update_id=1, text=OFF_TOPIC)
        return self._open_app()

    # -- App wiring -------------------------------------------------------------

    def test_queue_view_states_no_automatic_send(self, app):
        assert not app.exception
        assert any("açık onay" in warning.value for warning in app.warning)

    def test_queue_view_demo_has_no_queue_database_or_send_control(self, app, tmp_path):
        assert any("örnek veri" in info.value.lower() for info in app.info)
        assert SEND_LABEL not in self._labels(app)
        assert not (tmp_path / "data").exists()

    def test_queue_view_keeps_the_existing_views(self, app):
        assert list(app.sidebar.radio[0].options) == [
            "Asistan",
            "Gelen Mesaj Simülasyonu",
            "Onay Kuyruğu",
        ]

    def test_refusal_has_no_send_control(self, app_with_refusal):
        assert not app_with_refusal.exception
        assert not any(SEND_LABEL == b.label for b in app_with_refusal.button)
        assert REJECT_LABEL in self._labels(app_with_refusal)

    def test_queue_view_private_mode_hides_ids_and_paths(self, app_with_refusal, tmp_path):
        rendered = _rendered_text(app_with_refusal)
        assert OFF_TOPIC in rendered
        assert "Katılımcı#ab12" in rendered
        assert str(SYNTHETIC_CHAT_ID) not in rendered
        assert str(SYNTHETIC_CHAT_ID).lstrip("-") not in rendered
        assert str(tmp_path) not in rendered
        assert "config.json" not in rendered
        assert SYNTHETIC_ALIAS not in rendered

    def test_queue_view_without_token_reports_and_keeps_item_pending(
        self, private_workspace, draft_result
    ):
        item = self._enqueue(private_workspace, draft_result, update_id=2)
        at = self._open_app()
        at.checkbox(key=f"approval-confirm-{item.id}").check().run()
        at = at.button(key=f"approval-send-{item.id}").click().run()
        assert not at.exception
        assert any("token" in error.value.lower() for error in at.error)
        assert private_workspace.get(item.id).status is QueueStatus.PENDING

    def test_unreadable_private_config_falls_back_without_echoing_it(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        secret_path = tmp_path / "gizli-yapilandirma.json"
        secret_path.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv(PRIVATE_CONFIG_ENV, str(secret_path))
        at = self._open_app()
        assert not at.exception
        assert at.error
        assert "gizli-yapilandirma" not in _rendered_text(at)
        assert SEND_LABEL not in self._labels(at)

    # -- Queue rendering and actions ---------------------------------------------

    def test_queue_view_render_never_builds_a_transport(self, queue, draft_result):
        self._enqueue(queue, draft_result, update_id=3)
        factory = _TransportFactory(_FakeTransport())
        at = self._page(queue, factory)
        assert not at.exception
        assert factory.calls == 0
        assert factory.transport.sent == []

    def test_queue_view_pending_draft_shows_safe_metadata_and_editable_draft(
        self, queue, draft_result
    ):
        item = self._enqueue(queue, draft_result, update_id=4)
        at = self._page(queue, _TransportFactory(_FakeTransport()))
        rendered = _rendered_text(at)
        draft_box = at.text_area(key=f"approval-draft-{item.id}")
        assert draft_box.value == item.draft
        assert not draft_box.disabled
        assert "Katılımcı#ab12" in rendered
        assert "Telegram grubu#cd34" in rendered
        assert "Teslim kuralı" in rendered
        assert item.citations and item.citations[0].label in rendered
        assert str(SYNTHETIC_CHAT_ID) not in rendered
        assert str(item.message_id) not in rendered
        assert {SEND_LABEL, REJECT_LABEL} <= set(self._labels(at))

    def test_queue_view_send_requires_explicit_confirmation(self, queue, draft_result):
        item = self._enqueue(queue, draft_result, update_id=5)
        factory = _TransportFactory(_FakeTransport())
        at = self._page(queue, factory)
        assert at.button(key=f"approval-send-{item.id}").disabled
        at.button(key=f"approval-send-{item.id}").click().run()
        assert factory.calls == 0
        assert queue.get(item.id).status is QueueStatus.PENDING

    def test_queue_view_approval_sends_the_edited_draft_once(self, queue, draft_result):
        item = self._enqueue(queue, draft_result, update_id=6)
        factory = _TransportFactory(_FakeTransport())
        at = self._page(queue, factory)
        at.text_area(key=f"approval-draft-{item.id}").set_value("İnsan düzenledi.")
        at.checkbox(key=f"approval-confirm-{item.id}").check().run()
        at = at.button(key=f"approval-send-{item.id}").click().run()
        assert not at.exception
        assert factory.transport.sent == [(SYNTHETIC_CHAT_ID, "İnsan düzenledi.", item.message_id)]
        assert queue.get(item.id).status is QueueStatus.SENT
        assert any("gönderildi" in success.value for success in at.success)
        assert SEND_LABEL not in self._labels(at)
        at.run()
        assert len(factory.transport.sent) == 1

    def test_queue_view_reject_only_pending_and_never_sends(self, queue, draft_result):
        item = self._enqueue(queue, draft_result, update_id=7)
        factory = _TransportFactory(_FakeTransport())
        at = self._page(queue, factory)
        at = at.button(key=f"approval-reject-{item.id}").click().run()
        assert not at.exception
        assert queue.get(item.id).status is QueueStatus.REJECTED
        assert factory.calls == 0
        assert not {SEND_LABEL, REJECT_LABEL} & set(self._labels(at))

    def test_queue_view_refusal_can_be_rejected_but_not_sent(self, queue, refusal_result):
        item = self._enqueue(queue, refusal_result, update_id=8, text=OFF_TOPIC)
        at = self._page(queue, _TransportFactory(_FakeTransport()))
        assert SEND_LABEL not in self._labels(at)
        assert REJECT_LABEL in self._labels(at)
        assert not at.checkbox
        assert f"approval-draft-{item.id}" not in [area.key for area in at.text_area]

    def test_queue_view_failed_item_can_be_retried_but_not_rejected(
        self, queue, draft_result
    ):
        item = self._enqueue(queue, draft_result, update_id=9)
        queue.claim_for_send(item.id, item.draft)
        queue.mark_failed(item.id, "Telegram sendMessage failed (HTTP 400) chat not found")
        factory = _TransportFactory(_FakeTransport())
        at = self._page(queue, factory)
        assert SEND_LABEL in self._labels(at)
        assert REJECT_LABEL not in self._labels(at)
        assert any("chat not found" in error.value for error in at.error)
        at.checkbox(key=f"approval-confirm-{item.id}").check().run()
        at.button(key=f"approval-send-{item.id}").click().run()
        assert queue.get(item.id).status is QueueStatus.SENT
        assert len(factory.transport.sent) == 1

    def test_queue_view_rejected_delivery_becomes_failed(self, queue, draft_result):
        item = self._enqueue(queue, draft_result, update_id=10)
        factory = _TransportFactory(
            _FakeTransport(TelegramDeliveryRejected("Telegram sendMessage failed (HTTP 400)"))
        )
        at = self._page(queue, factory)
        at.checkbox(key=f"approval-confirm-{item.id}").check().run()
        at = at.button(key=f"approval-send-{item.id}").click().run()
        assert not at.exception
        assert queue.get(item.id).status is QueueStatus.FAILED
        assert any("HTTP 400" in error.value for error in at.error)

    def test_queue_view_uncertain_delivery_locks_the_item(self, queue, draft_result):
        item = self._enqueue(queue, draft_result, update_id=11)
        factory = _TransportFactory(
            _FakeTransport(TelegramDeliveryUncertain("Telegram sendMessage outcome unknown"))
        )
        at = self._page(queue, factory)
        at.checkbox(key=f"approval-confirm-{item.id}").check().run()
        at = at.button(key=f"approval-send-{item.id}").click().run()
        assert not at.exception
        assert queue.get(item.id).status is QueueStatus.SENDING
        assert not {SEND_LABEL, REJECT_LABEL} & set(self._labels(at))
        assert any("manuel" in warning.value.lower() for warning in at.warning)
        assert len(factory.transport.sent) == 1

    def test_queue_view_sending_row_is_locked_for_manual_review(self, queue, draft_result):
        item = self._enqueue(queue, draft_result, update_id=12)
        queue.claim_for_send(item.id, item.draft)
        factory = _TransportFactory(_FakeTransport())
        at = self._page(queue, factory)
        assert not {SEND_LABEL, REJECT_LABEL} & set(self._labels(at))
        assert at.text_area(key=f"approval-draft-{item.id}").disabled
        assert any("manuel" in warning.value.lower() for warning in at.warning)
        assert factory.calls == 0

    def test_queue_view_sent_item_is_read_only(self, queue, draft_result):
        item = self._enqueue(queue, draft_result, update_id=13)
        queue.claim_for_send(item.id, item.draft)
        queue.mark_sent(item.id)
        at = self._page(queue, _TransportFactory(_FakeTransport()))
        assert not {SEND_LABEL, REJECT_LABEL} & set(self._labels(at))
        assert at.text_area(key=f"approval-draft-{item.id}").disabled

    def test_queue_view_empty_queue_says_so(self, queue):
        at = self._page(queue, _TransportFactory(_FakeTransport()))
        assert not at.exception
        assert any("boş" in info.value for info in at.info)
