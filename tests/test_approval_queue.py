"""Real-SQLite tests for the durable, human-gated approval queue."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from staj_asistan.approval_queue import (
    ApprovalQueue,
    InvalidTransition,
    QueueStatus,
)
from staj_asistan.models import (
    Answer,
    AssistantMode,
    AuthorRole,
    ChunkCategory,
    Citation,
    EvidenceLevel,
)
from staj_asistan.triage import (
    IncomingMessage,
    MessageIntent,
    ReplyDecision,
    TriageResult,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "approval.sqlite3"


@pytest.fixture
def queue(db_path: Path) -> ApprovalQueue:
    return ApprovalQueue(db_path)


def _result(decision: ReplyDecision = ReplyDecision.DRAFT_REPLY) -> TriageResult:
    draft = "" if decision is ReplyDecision.DO_NOT_ANSWER else "Masked draft"
    citation = Citation(
        index=1,
        label="safe-source · Eğitmen · tarihsiz",
        source="safe-source",
        role=AuthorRole.INSTRUCTOR,
        snippet="Masked evidence",
        score=0.91,
    )
    return TriageResult(
        intent=MessageIntent.SUBMISSION_REQUIREMENT,
        selected_mode=AssistantMode.SUBMISSION_CHECKLIST,
        answer=Answer(
            text=draft,
            mode=AssistantMode.SUBMISSION_CHECKLIST,
            evidence=EvidenceLevel.HIGH,
            citations=[] if decision is ReplyDecision.DO_NOT_ANSWER else [citation],
        ),
        evidence_level=EvidenceLevel.HIGH,
        source_categories=(ChunkCategory.SUBMISSION,),
        reply_decision=decision,
        reason="Safe reason",
        draft_reply=draft,
        message=IncomingMessage(text="Masked incoming"),
    )


def _enqueue(
    queue: ApprovalQueue,
    *,
    update_id: int = 7,
    decision: ReplyDecision = ReplyDecision.DRAFT_REPLY,
):
    return queue.enqueue(
        update_id=update_id,
        chat_id=-100123,
        message_id=41,
        incoming_text="Masked incoming with a quote: '",
        sender_alias="Participant#safe1",
        group_label="Safe group",
        triage_result=_result(decision),
    )


def test_update_id_is_idempotent(queue: ApprovalQueue):
    first = _enqueue(queue)
    second = _enqueue(queue)

    assert first.id == second.id
    assert len(queue.list_items()) == 1


def test_queue_survives_reopen(db_path: Path):
    item = _enqueue(ApprovalQueue(db_path))

    reopened = ApprovalQueue(db_path)

    assert reopened.get(item.id) == item
    assert reopened.list_items() == [item]


def test_schema_uses_wal_and_unique_update_id(queue: ApprovalQueue, db_path: Path):
    _enqueue(queue)
    with sqlite3.connect(db_path) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        indexes = connection.execute("PRAGMA index_list(approval_items)").fetchall()

    assert mode.lower() == "wal"
    assert any(index[2] for index in indexes)


def test_triage_metadata_and_citations_round_trip(queue: ApprovalQueue):
    item = _enqueue(queue)

    assert item.intent is MessageIntent.SUBMISSION_REQUIREMENT
    assert item.selected_mode is AssistantMode.SUBMISSION_CHECKLIST
    assert item.evidence_level is EvidenceLevel.HIGH
    assert item.source_categories == (ChunkCategory.SUBMISSION,)
    assert item.reply_decision is ReplyDecision.DRAFT_REPLY
    assert item.reason == "Safe reason"
    assert item.draft == "Masked draft"
    assert len(item.citations) == 1
    assert item.citations[0].snippet == "Masked evidence"


def test_parameterized_values_round_trip_without_sql_effect(queue: ApprovalQueue):
    item = queue.enqueue(
        update_id=11,
        chat_id=-100123,
        message_id=42,
        incoming_text="Masked value'); DROP TABLE approval_items; --",
        sender_alias="Participant#safe2",
        group_label="Safe group's label",
        triage_result=_result(),
    )

    assert queue.get(item.id).incoming_text == "Masked value'); DROP TABLE approval_items; --"
    assert len(queue.list_items()) == 1


def test_do_not_answer_cannot_be_claimed(queue: ApprovalQueue):
    item = _enqueue(queue, update_id=8, decision=ReplyDecision.DO_NOT_ANSWER)

    with pytest.raises(InvalidTransition):
        queue.claim_for_send(item.id, edited_draft="anything")


def test_claim_is_atomic(queue: ApprovalQueue):
    item = _enqueue(queue, update_id=9)

    claimed = queue.claim_for_send(item.id, edited_draft="Reviewed")

    assert claimed.status is QueueStatus.SENDING
    assert claimed.draft == "Reviewed"
    with pytest.raises(InvalidTransition):
        ApprovalQueue(queue.db_path).claim_for_send(item.id, edited_draft="duplicate")


def test_empty_reviewed_draft_cannot_be_claimed(queue: ApprovalQueue):
    item = _enqueue(queue)

    with pytest.raises(ValueError, match="draft"):
        queue.claim_for_send(item.id, edited_draft="  ")


def test_reject_transitions_pending_item(queue: ApprovalQueue):
    item = _enqueue(queue)

    rejected = queue.reject(item.id)

    assert rejected.status is QueueStatus.REJECTED
    with pytest.raises(InvalidTransition):
        queue.claim_for_send(item.id, edited_draft="Reviewed")


def test_send_success_and_failure_transitions(queue: ApprovalQueue):
    sent_item = queue.claim_for_send(_enqueue(queue, update_id=20).id, "Reviewed")
    failed_item = queue.claim_for_send(_enqueue(queue, update_id=21).id, "Reviewed")

    assert queue.mark_sent(sent_item.id).status is QueueStatus.SENT
    failed = queue.mark_failed(failed_item.id, "safe timeout")
    assert failed.status is QueueStatus.FAILED
    assert failed.error == "safe timeout"


def test_failure_error_is_sanitized_and_retryable(queue: ApprovalQueue):
    item = queue.claim_for_send(_enqueue(queue, update_id=22).id, "Reviewed")
    fake_token = "123456789:" + ("A" * 30)
    unsafe_error = (
        f"line one\nrequest failed with {fake_token} at https://example.invalid/private "
        + ("x" * 600)
    )

    failed = queue.mark_failed(item.id, unsafe_error)
    retried = queue.claim_for_send(item.id, "Reviewed again")

    assert "\n" not in failed.error
    assert fake_token not in failed.error
    assert "example.invalid" not in failed.error
    assert len(failed.error) <= 240
    assert retried.status is QueueStatus.SENDING


def test_invalid_send_completion_does_not_mutate_pending_item(queue: ApprovalQueue):
    item = _enqueue(queue)

    with pytest.raises(InvalidTransition):
        queue.mark_sent(item.id)

    assert queue.get(item.id).status is QueueStatus.PENDING


def test_list_items_can_filter_status(queue: ApprovalQueue):
    pending = _enqueue(queue, update_id=30)
    rejected = queue.reject(_enqueue(queue, update_id=31).id)

    assert queue.list_items(status=QueueStatus.PENDING) == [pending]
    assert queue.list_items(status=QueueStatus.REJECTED) == [rejected]


def test_missing_item_is_reported(queue: ApprovalQueue):
    with pytest.raises(KeyError):
        queue.get(999)
