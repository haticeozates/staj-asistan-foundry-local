"""Offline tests for the Telegram boundary: transport, poller and approval service.

No test here may reach Telegram. HTTP behaviour is exercised through an injected
fake session and every other test uses an in-memory fake transport. The token and
all identities below are synthetic.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest
import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

from staj_asistan.approval_queue import ApprovalQueue, InvalidTransition, QueueStatus
from staj_asistan.models import (
    Answer,
    AssistantMode,
    AuthorRole,
    ChunkCategory,
    Citation,
    EvidenceLevel,
)
from staj_asistan.telegram import (
    TelegramDeliveryRejected,
    TelegramDeliveryUncertain,
    TelegramError,
    TelegramHTTPTransport,
)
from staj_asistan.telegram_worker import ApprovalService, TelegramPoller
from staj_asistan.triage import (
    Channel,
    IncomingMessage,
    MessageIntent,
    ReplyDecision,
    TriageResult,
    triage,
)

FAKE_TOKEN = "123456789:" + "Z" * 35
ALLOWED_CHAT = -100555
SENDER_ID = 987654321
SENDER_FIRST_NAME = "Zeynepfake"
SENDER_USERNAME = "zeynep_fake_handle"
CHAT_TITLE = "Gizli Staj Grubu Fake"


@pytest.fixture(autouse=True)
def _no_real_http(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("tests must never perform real HTTP requests")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeTransport:
    def __init__(self, updates=None, send_error: Exception | None = None, delay: float = 0.0):
        self.batches = list(updates or [])
        self.offsets: list[int | None] = []
        self.sent: list[tuple[int, str, int]] = []
        self.send_error = send_error
        self.delay = delay
        self._lock = threading.Lock()

    def get_updates(self, offset, timeout):
        self.offsets.append(offset)
        return self.batches.pop(0) if self.batches else []

    def send_message(self, chat_id, text, reply_to_message_id):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            if self.send_error is not None:
                raise self.send_error
            self.sent.append((chat_id, text, reply_to_message_id))
            return 5000 + len(self.sent)


class FakeAssistant:
    def __init__(self):
        self.questions: list[str] = []

    def ask(self, question, mode=AssistantMode.INSTRUCTOR_QA, top_k=None):
        self.questions.append(question)
        citation = Citation(
            index=1,
            label="safe-source · Eğitmen · tarihsiz",
            source="safe-source",
            role=AuthorRole.INSTRUCTOR,
            snippet="Masked evidence",
            score=0.9,
        )
        return Answer(
            text="Teslim linkini e-posta ile gönderin.",
            mode=mode,
            evidence=EvidenceLevel.HIGH,
            citations=[citation],
        )


class FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error=False):
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, response=None, error: Exception | None = None):
        self.calls: list[dict] = []
        self.response = response
        self.error = error

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        pass


def text_update(
    *,
    update_id: int = 100,
    chat_id: int = ALLOWED_CHAT,
    chat_type: str = "supergroup",
    text: str | None = "Teslim nasıl yapılır?",
    is_bot: bool = False,
    message_id: int = 77,
) -> dict:
    message: dict = {
        "message_id": message_id,
        "date": 1_790_000_000,
        "chat": {"id": chat_id, "type": chat_type, "title": CHAT_TITLE},
        "from": {
            "id": SENDER_ID,
            "is_bot": is_bot,
            "first_name": SENDER_FIRST_NAME,
            "username": SENDER_USERNAME,
        },
    }
    if text is not None:
        message["text"] = text
    return {"update_id": update_id, "message": message}


def _result(decision: ReplyDecision = ReplyDecision.DRAFT_REPLY) -> TriageResult:
    draft = "" if decision is ReplyDecision.DO_NOT_ANSWER else "Masked draft"
    return TriageResult(
        intent=MessageIntent.SUBMISSION_REQUIREMENT,
        selected_mode=AssistantMode.SUBMISSION_CHECKLIST,
        answer=Answer(
            text=draft,
            mode=AssistantMode.SUBMISSION_CHECKLIST,
            evidence=EvidenceLevel.HIGH,
        ),
        evidence_level=EvidenceLevel.HIGH,
        source_categories=(ChunkCategory.SUBMISSION,),
        reply_decision=decision,
        reason="Safe reason",
        draft_reply=draft,
    )


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "queue.sqlite3"


@pytest.fixture
def queue(db_path: Path) -> ApprovalQueue:
    return ApprovalQueue(db_path)


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def assistant() -> FakeAssistant:
    return FakeAssistant()


@pytest.fixture
def poller(queue, fake_transport, assistant) -> TelegramPoller:
    return TelegramPoller(
        queue=queue,
        transport=fake_transport,
        assistant=assistant,
        allowed_chat_ids={ALLOWED_CHAT},
    )


@pytest.fixture
def service(queue, fake_transport) -> ApprovalService:
    return ApprovalService(queue=queue, transport=fake_transport)


def _enqueue(queue: ApprovalQueue, update_id: int = 10, decision=ReplyDecision.DRAFT_REPLY):
    return queue.enqueue(
        update_id=update_id,
        chat_id=ALLOWED_CHAT,
        message_id=77,
        incoming_text="Masked incoming",
        sender_alias="Katılımcı#safe",
        group_label="Telegram grubu#safe",
        triage_result=_result(decision),
    )


# --------------------------------------------------------------------------------------
# HTTP transport
# --------------------------------------------------------------------------------------


def test_from_env_reads_token_only_from_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    session = FakeSession(FakeResponse(payload={"ok": True, "result": []}))

    transport = TelegramHTTPTransport.from_env(session=session)
    transport.get_updates(offset=None, timeout=0)

    assert FAKE_TOKEN in session.calls[0]["url"]
    assert FAKE_TOKEN not in repr(transport)
    assert FAKE_TOKEN not in str(transport)


def test_from_env_fails_closed_without_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(TelegramError, match="TELEGRAM_BOT_TOKEN"):
        TelegramHTTPTransport.from_env(session=FakeSession())


def test_blank_env_token_is_rejected(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "   ")

    with pytest.raises(TelegramError):
        TelegramHTTPTransport.from_env(session=FakeSession())


def test_get_updates_uses_long_poll_offset_and_bounded_timeouts():
    session = FakeSession(FakeResponse(payload={"ok": True, "result": [{"update_id": 3}]}))
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=session, connect_timeout=4.0)

    updates = transport.get_updates(offset=42, timeout=25)

    call = session.calls[0]
    assert updates == [{"update_id": 3}]
    assert call["url"].endswith("/getUpdates")
    assert call["json"]["offset"] == 42
    assert call["json"]["timeout"] == 25
    assert call["json"]["allowed_updates"] == ["message"]
    connect, read = call["timeout"]
    assert connect == 4.0
    assert 25 < read <= 25 + 60


def test_get_updates_without_offset_omits_it():
    session = FakeSession(FakeResponse(payload={"ok": True, "result": []}))
    TelegramHTTPTransport(FAKE_TOKEN, session=session).get_updates(offset=None, timeout=0)

    assert "offset" not in session.calls[0]["json"]


def test_send_message_posts_reviewed_text_as_reply():
    session = FakeSession(
        FakeResponse(payload={"ok": True, "result": {"message_id": 901}})
    )
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=session)

    sent_id = transport.send_message(ALLOWED_CHAT, "Reviewed text", 77)

    call = session.calls[0]
    assert sent_id == 901
    assert call["url"].endswith("/sendMessage")
    assert call["json"]["chat_id"] == ALLOWED_CHAT
    assert call["json"]["text"] == "Reviewed text"
    assert call["json"]["reply_parameters"] == {"message_id": 77}
    assert all(isinstance(part, float) and part > 0 for part in call["timeout"])


def _assert_token_free(exc: BaseException) -> None:
    assert FAKE_TOKEN not in str(exc)
    assert "api.telegram.org" not in str(exc)
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True


def _unsafe_message() -> str:
    return f"HTTPSConnectionPool: Max retries exceeded with url: /bot{FAKE_TOKEN}/sendMessage"


def test_read_timeout_on_send_is_uncertain_and_token_free():
    session = FakeSession(error=requests.ReadTimeout(_unsafe_message()))
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=session)

    with pytest.raises(TelegramDeliveryUncertain) as caught:
        transport.send_message(ALLOWED_CHAT, "text", 77)

    _assert_token_free(caught.value)


def test_connection_reset_on_send_is_uncertain():
    session = FakeSession(error=requests.ConnectionError(_unsafe_message()))
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=session)

    with pytest.raises(TelegramDeliveryUncertain) as caught:
        transport.send_message(ALLOWED_CHAT, "text", 77)

    _assert_token_free(caught.value)


def test_connect_timeout_on_send_is_a_definite_rejection():
    session = FakeSession(error=requests.ConnectTimeout(_unsafe_message()))
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=session)

    with pytest.raises(TelegramDeliveryRejected) as caught:
        transport.send_message(ALLOWED_CHAT, "text", 77)

    _assert_token_free(caught.value)


def test_connection_refused_before_request_is_a_definite_rejection():
    reason = NewConnectionError(None, "refused")
    wrapped = MaxRetryError(None, f"/bot{FAKE_TOKEN}/sendMessage", reason)
    session = FakeSession(error=requests.ConnectionError(wrapped))
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=session)

    with pytest.raises(TelegramDeliveryRejected) as caught:
        transport.send_message(ALLOWED_CHAT, "text", 77)

    _assert_token_free(caught.value)


def test_api_client_error_is_a_definite_rejection_with_sanitized_description():
    payload = {
        "ok": False,
        "error_code": 400,
        "description": f"Bad Request: chat not found bot{FAKE_TOKEN}",
    }
    transport = TelegramHTTPTransport(
        FAKE_TOKEN, session=FakeSession(FakeResponse(400, payload))
    )

    with pytest.raises(TelegramDeliveryRejected, match="400") as caught:
        transport.send_message(ALLOWED_CHAT, "text", 77)

    _assert_token_free(caught.value)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(502, {"ok": False, "error_code": 502, "description": "Bad Gateway"}),
        FakeResponse(200, json_error=True),
        FakeResponse(200, {"ok": True, "result": {}}),
    ],
    ids=["server-error", "unparseable-body", "missing-message-id"],
)
def test_ambiguous_send_responses_are_uncertain(response):
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=FakeSession(response))

    with pytest.raises(TelegramDeliveryUncertain):
        transport.send_message(ALLOWED_CHAT, "text", 77)


def test_get_updates_failure_is_sanitized():
    session = FakeSession(error=requests.ReadTimeout(_unsafe_message()))
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=session)

    with pytest.raises(TelegramError) as caught:
        transport.get_updates(offset=1, timeout=1)

    _assert_token_free(caught.value)


def test_get_updates_api_error_is_sanitized():
    payload = {"ok": False, "error_code": 401, "description": f"Unauthorized {FAKE_TOKEN}"}
    transport = TelegramHTTPTransport(FAKE_TOKEN, session=FakeSession(FakeResponse(401, payload)))

    with pytest.raises(TelegramError) as caught:
        transport.get_updates(offset=1, timeout=1)

    _assert_token_free(caught.value)


def test_http_client_debug_logs_never_contain_the_token(caplog):
    TelegramHTTPTransport(FAKE_TOKEN, session=FakeSession())
    with caplog.at_level(logging.DEBUG, logger="urllib3.connectionpool"):
        logging.getLogger("urllib3.connectionpool").debug(
            '%s "POST /bot%s/getUpdates HTTP/1.1" 200', "https://api.telegram.org:443", FAKE_TOKEN
        )

    assert caplog.records
    assert FAKE_TOKEN not in caplog.text


# --------------------------------------------------------------------------------------
# Poller
# --------------------------------------------------------------------------------------


def test_poller_requires_an_allowlist(queue, fake_transport, assistant):
    with pytest.raises(ValueError, match="allowed"):
        TelegramPoller(
            queue=queue, transport=fake_transport, assistant=assistant, allowed_chat_ids=set()
        )


def test_unallowed_chat_is_ignored(poller, assistant):
    assert poller.process_update(text_update(chat_id=999)) is None
    assert poller.queue.list_items() == []
    assert assistant.questions == []


@pytest.mark.parametrize(
    "update",
    [
        text_update(chat_type="private"),
        text_update(chat_type="channel"),
        text_update(is_bot=True),
        text_update(text=None),
        text_update(text="   "),
        {"update_id": 100, "edited_message": text_update()["message"]},
        {"update_id": 100, "message": {**text_update()["message"], "from": None}},
    ],
    ids=["private", "channel", "bot", "non-text", "blank", "edited", "no-sender"],
)
def test_unsupported_updates_are_ignored(poller, assistant, update):
    assert poller.process_update(update) is None
    assert poller.queue.list_items() == []
    assert assistant.questions == []


def test_message_is_masked_and_sender_is_pseudonymous(poller):
    poller.process_update(text_update(text="mail me at real@private.test"))

    item = poller.queue.list_items()[0]

    assert "real@private.test" not in item.incoming_text
    assert "[E-POSTA]" in item.incoming_text
    assert item.sender_alias.startswith("Katılımcı#")


def test_triage_only_sees_masked_text(poller, assistant):
    poller.process_update(text_update(text="teslim için real@private.test adresine yazın"))

    assert assistant.questions
    assert all("real@private.test" not in question for question in assistant.questions)


def test_incoming_message_is_telegram_channel(queue, fake_transport, assistant):
    seen: list[IncomingMessage] = []

    def spy(message, assistant_):
        seen.append(message)
        return triage(message, assistant_)

    poller = TelegramPoller(
        queue=queue,
        transport=fake_transport,
        assistant=assistant,
        allowed_chat_ids={ALLOWED_CHAT},
        triage_fn=spy,
    )
    poller.process_update(text_update())

    assert seen[0].channel is Channel.TELEGRAM
    assert seen[0].sender_alias.startswith("Katılımcı#")
    assert seen[0].received_at is not None


def test_raw_profile_identity_and_chat_title_are_never_persisted(poller, db_path):
    poller.process_update(text_update(text="mail me at real@private.test"))

    item = poller.queue.list_items()[0]
    blob = b"".join(
        path.read_bytes()
        for path in db_path.parent.iterdir()
        if path.name.startswith(db_path.name)
    )

    for secret in (SENDER_FIRST_NAME, SENDER_USERNAME, str(SENDER_ID), CHAT_TITLE, "real@private"):
        assert secret.encode() not in blob
        assert secret not in item.sender_alias
        assert secret not in item.group_label
    assert item.group_label.startswith("Telegram grubu#")
    assert item.chat_id == ALLOWED_CHAT
    assert item.message_id == 77


def test_sender_pseudonym_is_stable_per_sender(poller):
    first = poller.queue.get(poller.process_update(text_update(update_id=1)))
    second = poller.queue.get(poller.process_update(text_update(update_id=2)))

    assert first.sender_alias == second.sender_alias


def test_duplicate_update_creates_one_row_and_triages_once(poller, assistant):
    first = poller.process_update(text_update(update_id=100))
    second = poller.process_update(text_update(update_id=100))

    assert first == second
    assert len(poller.queue.list_items()) == 1
    assert len(assistant.questions) == 1


def test_duplicate_update_after_restart_is_not_retriaged(queue, fake_transport, assistant):
    kwargs = dict(
        queue=queue, transport=fake_transport, assistant=assistant, allowed_chat_ids={ALLOWED_CHAT}
    )
    first = TelegramPoller(**kwargs).process_update(text_update(update_id=100))
    asked = len(assistant.questions)

    second = TelegramPoller(**kwargs).process_update(text_update(update_id=100))

    assert first == second
    assert len(assistant.questions) == asked
    assert len(queue.list_items()) == 1


def test_offset_advances_past_ignored_and_accepted_updates(queue, assistant):
    transport = FakeTransport(
        updates=[[text_update(update_id=5, chat_id=999), text_update(update_id=6)], []]
    )
    poller = TelegramPoller(
        queue=queue, transport=transport, assistant=assistant, allowed_chat_ids={ALLOWED_CHAT}
    )

    created = poller.poll_once(timeout=0)
    poller.poll_once(timeout=0)

    assert created == 1
    assert poller.offset == 7
    assert transport.offsets == [None, 7]


def test_offset_never_moves_backwards(poller):
    poller.process_update(text_update(update_id=20, chat_id=999))
    poller.process_update(text_update(update_id=3, chat_id=999))

    assert poller.offset == 21


def test_failed_triage_keeps_offset_so_update_is_retried(queue, assistant):
    batch = [text_update(update_id=8), text_update(update_id=9)]
    transport = FakeTransport(updates=[batch, batch])
    calls = {"n": 0}

    def flaky(message, assistant_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model server down")
        return triage(message, assistant_)

    poller = TelegramPoller(
        queue=queue,
        transport=transport,
        assistant=assistant,
        allowed_chat_ids={ALLOWED_CHAT},
        triage_fn=flaky,
    )

    with pytest.raises(RuntimeError):
        poller.poll_once(timeout=0)

    assert poller.offset is None
    assert queue.list_items() == []

    assert poller.poll_once(timeout=0) == 2
    assert transport.offsets == [None, None]
    assert poller.offset == 10


def test_polling_never_sends(poller, fake_transport):
    fake_transport.batches = [[text_update(update_id=1), text_update(update_id=2)]]

    poller.poll_once(timeout=0)

    assert len(poller.queue.list_items(status=QueueStatus.PENDING)) == 2
    assert fake_transport.sent == []


def test_run_forever_backs_off_after_polling_failure(queue, assistant, caplog):
    stop = threading.Event()

    class FlakyTransport(FakeTransport):
        def get_updates(self, offset, timeout):
            self.offsets.append(offset)
            if len(self.offsets) == 1:
                raise TelegramError("Telegram getUpdates failed (ReadTimeout)")
            stop.set()
            return [text_update(update_id=1)]

    transport = FlakyTransport()
    poller = TelegramPoller(
        queue=queue, transport=transport, assistant=assistant, allowed_chat_ids={ALLOWED_CHAT}
    )

    with caplog.at_level(logging.WARNING, logger="staj_asistan"):
        poller.run_forever(stop, timeout=0, backoff_seconds=0)

    assert len(transport.offsets) == 2
    assert len(queue.list_items()) == 1
    assert "ReadTimeout" in caplog.text
    assert transport.sent == []


def test_poller_log_output_has_no_message_content(poller, caplog):
    with caplog.at_level(logging.DEBUG, logger="staj_asistan"):
        poller.process_update(text_update(update_id=1, text="real@private.test yardım"))
        poller.process_update(text_update(update_id=2, chat_id=999, text="secret words"))

    for secret in ("real@private", "secret words", SENDER_FIRST_NAME, str(SENDER_ID), CHAT_TITLE):
        assert secret not in caplog.text
    assert str(ALLOWED_CHAT) not in caplog.text


# --------------------------------------------------------------------------------------
# Approval service
# --------------------------------------------------------------------------------------


def test_approval_sends_exactly_once(service, queue, fake_transport):
    item = _enqueue(queue)

    sent = service.approve_and_send(item.id, "Human edited draft")

    assert sent.status is QueueStatus.SENT
    assert fake_transport.sent == [(item.chat_id, "Human edited draft", item.message_id)]
    with pytest.raises(InvalidTransition):
        service.approve_and_send(item.id, "again")
    assert len(fake_transport.sent) == 1


def test_do_not_answer_can_never_send(service, queue, fake_transport):
    item = _enqueue(queue, decision=ReplyDecision.DO_NOT_ANSWER)

    with pytest.raises(InvalidTransition):
        service.approve_and_send(item.id, "Anything at all")

    assert fake_transport.sent == []
    assert queue.get(item.id).status is QueueStatus.PENDING


def test_rejected_item_cannot_send(service, queue, fake_transport):
    item = _enqueue(queue)
    queue.reject(item.id)

    with pytest.raises(InvalidTransition):
        service.approve_and_send(item.id, "Reviewed")

    assert fake_transport.sent == []


def test_overlong_draft_is_refused_before_claim(service, queue, fake_transport):
    item = _enqueue(queue)

    with pytest.raises(ValueError):
        service.approve_and_send(item.id, "x" * 4097)

    assert fake_transport.sent == []
    assert queue.get(item.id).status is QueueStatus.PENDING


def test_definite_failure_is_failed_and_needs_explicit_retry(queue, fake_transport):
    fake_transport.send_error = TelegramDeliveryRejected(f"HTTP 400 bot{FAKE_TOKEN}")
    service = ApprovalService(queue=queue, transport=fake_transport)
    item = _enqueue(queue)

    failed = service.approve_and_send(item.id, "Reviewed")

    assert failed.status is QueueStatus.FAILED
    assert FAKE_TOKEN not in failed.error
    assert fake_transport.sent == []

    fake_transport.send_error = None
    retried = service.approve_and_send(item.id, "Reviewed again")

    assert retried.status is QueueStatus.SENT
    assert fake_transport.sent == [(item.chat_id, "Reviewed again", item.message_id)]


@pytest.mark.parametrize(
    "error",
    [TelegramDeliveryUncertain("read timed out"), RuntimeError(f"boom {FAKE_TOKEN}")],
    ids=["uncertain", "unexpected"],
)
def test_ambiguous_failure_stays_sending_and_cannot_be_retried(queue, fake_transport, error):
    fake_transport.send_error = error
    service = ApprovalService(queue=queue, transport=fake_transport)
    item = _enqueue(queue)

    with pytest.raises(TelegramDeliveryUncertain) as caught:
        service.approve_and_send(item.id, "Reviewed")

    assert FAKE_TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None
    assert queue.get(item.id).status is QueueStatus.SENDING

    fake_transport.send_error = None
    with pytest.raises(InvalidTransition):
        service.approve_and_send(item.id, "Reviewed again")
    assert fake_transport.sent == []


def test_concurrent_approvals_send_once(db_path):
    transport = FakeTransport(delay=0.05)
    item = _enqueue(ApprovalQueue(db_path))
    outcomes: list[str] = []
    barrier = threading.Barrier(4)

    def approve():
        service = ApprovalService(queue=ApprovalQueue(db_path), transport=transport)
        barrier.wait()
        try:
            service.approve_and_send(item.id, "Reviewed")
            outcomes.append("sent")
        except InvalidTransition:
            outcomes.append("blocked")

    threads = [threading.Thread(target=approve) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["blocked", "blocked", "blocked", "sent"]
    assert len(transport.sent) == 1
