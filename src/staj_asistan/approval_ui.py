"""Streamlit approval queue: the only screen from which a reply can reach Telegram.

Rendering is side-effect free. A Telegram transport is built only inside the
callback of the explicitly labelled send button, after the approver has ticked a
confirmation box, and every send goes through :class:`ApprovalService`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import streamlit as st

from .approval_queue import ApprovalItem, ApprovalQueue, InvalidTransition, QueueStatus
from .models import AssistantMode, EvidenceLevel
from .privacy import mask_text
from .telegram import (
    MAX_MESSAGE_LENGTH,
    TelegramDeliveryUncertain,
    TelegramError,
    TelegramHTTPTransport,
    TelegramTransport,
)
from .telegram_worker import ApprovalService
from .triage import MessageIntent, ReplyDecision

TransportFactory = Callable[[], TelegramTransport]

VIEW_TITLE = "Onay Kuyruğu"
SEND_LABEL = "Onayla ve Telegram'a gönder"
REJECT_LABEL = "Reddet"
CONFIRM_LABEL = "Taslağı okudum; bu metnin Telegram grubuna gönderilmesini onaylıyorum."

NO_AUTO_SEND_NOTICE = (
    "Hiçbir cevap otomatik gönderilmez. Telegram'a giden her mesaj, bir insanın bu ekranda "
    "verdiği açık onayla ve tek tek gönderilir."
)
DEMO_NOTICE = (
    "Özel yapılandırma yok: uygulama örnek veri demosunda çalışıyor. Onay kuyruğu ve "
    "Telegram gönderimi yalnızca yerel özel yapılandırma ile açılır."
)
EMPTY_NOTICE = "Kuyruk boş: onay bekleyen Telegram mesajı yok."
LOCKED_NOTICE = (
    "Gönderim sonucu belirsiz: mesaj gruba ulaşmış olabilir. Öğe kilitlendi ve yeniden "
    "gönderilemez; Telegram grubunu kontrol ederek manuel inceleme yap."
)
REFUSAL_NOTICE = "Bu mesaj cevaplanmaz ve gönderilemez. Gerekirse eğitmene yönlendir."
MISSING_TOKEN_NOTICE = (
    "Telegram gönderimi kapalı: bu oturumda bot token'ı tanımlı değil. Öğe beklemede kaldı."
)

STATUS_LABELS: dict[QueueStatus, str] = {
    QueueStatus.PENDING: "Onay bekliyor",
    QueueStatus.SENDING: "Kilitli - manuel inceleme",
    QueueStatus.SENT: "Gönderildi",
    QueueStatus.FAILED: "Gönderilemedi - yeniden onaylanabilir",
    QueueStatus.REJECTED: "Reddedildi",
}

DECISION_LABELS: dict[ReplyDecision, str] = {
    ReplyDecision.DRAFT_REPLY: "Taslak hazır",
    ReplyDecision.NEEDS_HUMAN_APPROVAL: "İnsan onayı gerekiyor",
    ReplyDecision.DO_NOT_ANSWER: "Cevaplanmaz",
}

_OPEN_STATUSES = (QueueStatus.PENDING, QueueStatus.FAILED, QueueStatus.SENDING)
_FLASH_KEY = "approval_flash"


def can_send(item: ApprovalItem) -> bool:
    return (
        item.status in (QueueStatus.PENDING, QueueStatus.FAILED)
        and item.reply_decision is not ReplyDecision.DO_NOT_ANSWER
    )


def can_reject(item: ApprovalItem) -> bool:
    return item.status is QueueStatus.PENDING


def _keys(item_id: int) -> tuple[str, str]:
    return f"approval-draft-{item_id}", f"approval-confirm-{item_id}"


def _flash(kind: str, message: str) -> None:
    st.session_state[_FLASH_KEY] = (kind, message)


def _approve(queue: ApprovalQueue, transport_factory: TransportFactory, item_id: int) -> None:
    draft_key, confirm_key = _keys(item_id)
    confirmed = st.session_state.get(confirm_key, False)
    st.session_state[confirm_key] = False
    if not confirmed:
        _flash("warning", "Göndermeden önce onay kutusunu işaretle.")
        return
    draft = str(st.session_state.get(draft_key, ""))
    if not draft.strip() or len(draft) > MAX_MESSAGE_LENGTH:
        _flash("error", f"Taslak boş olamaz ve en fazla {MAX_MESSAGE_LENGTH} karakter olabilir.")
        return

    try:
        transport = transport_factory()
    except TelegramError:
        _flash("error", MISSING_TOKEN_NOTICE)
        return
    try:
        item = ApprovalService(queue=queue, transport=transport).approve_and_send(item_id, draft)
    except TelegramDeliveryUncertain:
        _flash("warning", f"#{item_id}: {LOCKED_NOTICE}")
    except (InvalidTransition, KeyError):
        _flash("warning", f"#{item_id} artık gönderilemez; durumu başka bir işlemle değişmiş.")
    except ValueError:
        _flash("error", f"Taslak boş olamaz ve en fazla {MAX_MESSAGE_LENGTH} karakter olabilir.")
    else:
        if item.status is QueueStatus.SENT:
            _flash("success", f"#{item_id} Telegram'a gönderildi.")
        else:
            _flash("error", f"#{item_id} gönderilemedi: {item.error}")
    finally:
        close = getattr(transport, "close", None)
        if callable(close):
            close()


def _reject(queue: ApprovalQueue, item_id: int) -> None:
    try:
        queue.reject(item_id)
    except (InvalidTransition, KeyError):
        _flash("warning", f"#{item_id} artık reddedilemez; durumu başka bir işlemle değişmiş.")
    else:
        _flash("info", f"#{item_id} reddedildi; gönderilmeyecek.")


def _label(labels: Mapping, value: MessageIntent | AssistantMode | EvidenceLevel) -> str:
    return labels.get(value, value.value)


def _render_item(
    item: ApprovalItem,
    queue: ApprovalQueue,
    transport_factory: TransportFactory,
    intent_labels: Mapping[MessageIntent, str],
    mode_labels: Mapping[AssistantMode, str],
) -> None:
    draft_key, confirm_key = _keys(item.id)
    with st.container(border=True):
        st.markdown(
            f"**#{item.id}** · {STATUS_LABELS[item.status]} · "
            f"{DECISION_LABELS[item.reply_decision]}"
        )
        st.caption(
            f"{item.sender_alias} · {item.group_label} · "
            f"{item.created_at.strftime('%d.%m.%Y %H:%M')} UTC"
        )
        st.markdown("**Gelen mesaj (maskeli)**")
        st.text(mask_text(item.incoming_text).text)

        left, mid, right = st.columns(3)
        left.metric("Niyet", _label(intent_labels, item.intent))
        mid.metric("Seçilen mod", _label(mode_labels, item.selected_mode))
        right.metric("Kanıt", item.evidence_level.value)
        st.caption(item.reason)
        if item.source_categories:
            st.caption(
                "Kaynak kategorileri: " + ", ".join(c.value for c in item.source_categories)
            )
        if item.citations:
            with st.expander(f"Kaynaklar ({len(item.citations)})"):
                for citation in item.citations:
                    st.markdown(
                        f"[{citation.index}] {citation.label} · skor {citation.score:.2f}"
                    )
                    st.text(mask_text(citation.snippet).text)

        if item.status is QueueStatus.FAILED and item.error:
            st.error(f"Son gönderim denemesi başarısız: {item.error}")
        if item.status is QueueStatus.SENDING:
            st.warning(LOCKED_NOTICE)

        sendable = can_send(item)
        if item.reply_decision is ReplyDecision.DO_NOT_ANSWER:
            st.info(REFUSAL_NOTICE)
        else:
            st.text_area(
                "Cevap taslağı (göndermeden önce düzenleyebilirsin)",
                value=item.draft,
                key=draft_key,
                height=160,
                max_chars=MAX_MESSAGE_LENGTH,
                disabled=not sendable,
            )

        if sendable:
            confirmed = st.checkbox(CONFIRM_LABEL, key=confirm_key)
            st.button(
                SEND_LABEL,
                key=f"approval-send-{item.id}",
                type="primary",
                disabled=not confirmed,
                on_click=_approve,
                args=(queue, transport_factory, item.id),
            )
        if can_reject(item):
            st.button(
                REJECT_LABEL,
                key=f"approval-reject-{item.id}",
                on_click=_reject,
                args=(queue, item.id),
            )


def render_approval_view(
    queue: ApprovalQueue | None,
    transport_factory: TransportFactory = TelegramHTTPTransport.from_env,
    *,
    intent_labels: Mapping[MessageIntent, str] | None = None,
    mode_labels: Mapping[AssistantMode, str] | None = None,
) -> None:
    """Render the queue; ``queue=None`` is the sample-data demo with no queue at all."""

    st.header(VIEW_TITLE)
    st.warning(NO_AUTO_SEND_NOTICE)
    if queue is None:
        st.info(DEMO_NOTICE)
        return

    flash = st.session_state.pop(_FLASH_KEY, None)
    if flash is not None:
        kind, message = flash
        getattr(st, kind)(message)

    items = queue.list_items()
    if not items:
        st.info(EMPTY_NOTICE)
        return

    counts = {status: sum(item.status is status for item in items) for status in QueueStatus}
    st.caption(
        " · ".join(f"{STATUS_LABELS[status]}: {count}" for status, count in counts.items())
    )
    open_items = [item for item in items if item.status in _OPEN_STATUSES]
    closed_items = [item for item in items if item.status not in _OPEN_STATUSES]
    labels = (intent_labels or {}, mode_labels or {})

    st.subheader("Açık öğeler")
    if not open_items:
        st.caption("Onay bekleyen öğe yok.")
    for item in open_items:
        _render_item(item, queue, transport_factory, *labels)

    if closed_items:
        st.subheader("Kapanan öğeler")
        for item in closed_items:
            _render_item(item, queue, transport_factory, *labels)
