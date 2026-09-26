"""Streamlit UI for StajAsistan 2.0.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from staj_asistan.models import AssistantMode, EvidenceLevel  # noqa: E402
from staj_asistan.pipeline import Assistant  # noqa: E402
from staj_asistan.triage import (  # noqa: E402
    Channel,
    IncomingMessage,
    MessageIntent,
    ReplyDecision,
    triage,
)

# The simulator is a view, not a fifth AssistantMode: it picks one of the four
# existing modes for you rather than adding a new answering style.
VIEW_ASSISTANT = "Asistan"
VIEW_TRIAGE = "Gelen Mesaj Simülasyonu"

MODE_LABELS: dict[AssistantMode, str] = {
    AssistantMode.INSTRUCTOR_QA: "Eğitmen Soru-Cevap",
    AssistantMode.SUBMISSION_CHECKLIST: "Teslim Kontrol Listesi",
    AssistantMode.CORRECTION_ANALYZER: "Düzeltme İsteği Analizi",
    AssistantMode.TECHNICAL_HELP: "Teknik Yardım",
}

MODE_PLACEHOLDERS: dict[AssistantMode, str] = {
    AssistantMode.INSTRUCTOR_QA: "Örnek: Final tesliminde ne gerekiyor?",
    AssistantMode.SUBMISSION_CHECKLIST: "Örnek: GitHub repo hazır, README var ama video çekmedim.",
    AssistantMode.CORRECTION_ANALYZER: "Örnek: Listede adım yanlış, projem Foundry Local olmalı, devam etmek istiyorum.",
    AssistantMode.TECHNICAL_HELP: "Örnek: Foundry Local çalışmazsa ne kontrol etmeliyim?",
}

EVIDENCE_BADGES: dict[EvidenceLevel, tuple[str, str]] = {
    EvidenceLevel.HIGH: ("Kanıt düzeyi: yüksek", "success"),
    EvidenceLevel.MEDIUM: ("Kanıt düzeyi: orta", "info"),
    EvidenceLevel.LOW: ("Kanıt düzeyi: düşük - cevabı kaynaklardan doğrula", "warning"),
    EvidenceLevel.NONE: ("Kanıt yok - kaynaklarda karşılık bulunamadı", "error"),
}

DEMO_QUESTIONS: list[tuple[str, AssistantMode]] = [
    ("Final tesliminde ne gerekiyor?", AssistantMode.SUBMISSION_CHECKLIST),
    ("Liste güncel değilse çalışmaya devam etmeli miyim?", AssistantMode.INSTRUCTOR_QA),
    (
        "GitHub repo hazır ama video çekmedim, teslim için eksiğim var mı?",
        AssistantMode.SUBMISSION_CHECKLIST,
    ),
    (
        "Listede projem yanlış görünüyor, Foundry Local olarak güncellenmesini istiyorum. Bu mesaj yeterli mi?",
        AssistantMode.CORRECTION_ANALYZER,
    ),
]

CHANNEL_LABELS: dict[Channel, str] = {
    Channel.WHATSAPP: "WhatsApp grubu",
    Channel.TELEGRAM: "Telegram grubu",
    Channel.MANUAL: "Elle giriş",
}

INTENT_LABELS: dict[MessageIntent, str] = {
    MessageIntent.SUBMISSION_REQUIREMENT: "Teslim kuralı / durumu",
    MessageIntent.SUBMISSION_CHANNEL: "Teslim kanalı",
    MessageIntent.ROSTER_CONTINUE: "Liste / devam durumu",
    MessageIntent.CORRECTION_REQUEST: "Düzeltme isteği",
    MessageIntent.TECHNICAL_HELP: "Teknik yardım",
    MessageIntent.CERTIFICATE: "Sertifika",
    MessageIntent.UNKNOWN: "Sınıflandırılamadı",
}

DECISION_BADGES: dict[ReplyDecision, tuple[str, str]] = {
    ReplyDecision.DRAFT_REPLY: ("Taslak hazır - insan kopyalayıp gönderir", "success"),
    ReplyDecision.NEEDS_HUMAN_APPROVAL: ("İnsan onayı gerekiyor", "warning"),
    ReplyDecision.DO_NOT_ANSWER: ("Cevaplama - eğitmene yönlendir", "error"),
}

SIMULATION_NOTICE = (
    "Bu bir simülasyondur. Sistem hiçbir kanala otomatik mesaj göndermez; "
    "sadece insan onayı için cevap taslağı üretir."
)

INCOMING_SAMPLES: list[str] = [
    "GitHub repo hazır ama video çekmedim, teslim olur mu?",
    "Listede projem yanlış görünüyor, Foundry Local olarak güncellenmesini istiyorum.",
    "WhatsApp'tan yazmam yeterli mi?",
    "Sertifikam ne zaman gelir?",
    "İstanbul'da hava nasıl?",
]


st.set_page_config(page_title="StajAsistan 2.0", page_icon="📘", layout="wide")


def get_assistant() -> Assistant:
    if "assistant" not in st.session_state:
        with st.spinner("Yerel modeller ve indeks hazırlanıyor…"):
            st.session_state.assistant = Assistant()
    return st.session_state.assistant


def render_sidebar(assistant: Assistant) -> tuple[str, AssistantMode]:
    with st.sidebar:
        st.title("StajAsistan 2.0")
        st.caption("AI Innovators grup operasyon asistanı")

        view = st.radio(
            "Görünüm",
            options=[VIEW_ASSISTANT, VIEW_TRIAGE],
            horizontal=True,
        )

        st.subheader("Veri")
        uploads = st.file_uploader(
            "WhatsApp dışa aktarımı veya not dosyası",
            type=["txt", "md", "log", "zip"],
            accept_multiple_files=True,
            help="Zip içindeki yalnızca metin dosyaları okunur; görsel ve ses dosyalarına hiç dokunulmaz.",
        )
        if uploads and st.button("Yüklenen dosyaları indeksle", use_container_width=True):
            added = 0
            for upload in uploads:
                added += assistant.ingest_upload(upload.getvalue(), upload.name)
            st.success(f"{added} chunk indekslendi.")

        col_a, col_b = st.columns(2)
        if col_a.button("Örnek veri", use_container_width=True):
            assistant.load_samples()
            st.success("Örnek veri indekslendi.")
        if col_b.button("Temizle", use_container_width=True):
            assistant.reset()
            st.session_state.pop("answer", None)
            st.info("İndeks temizlendi.")

        st.subheader("İndeks durumu")
        stats = assistant.stats
        left, right = st.columns(2)
        left.metric("Dosya", stats.file_count)
        right.metric("Mesaj", stats.message_count)
        left.metric("Eğitmen mesajı", stats.instructor_message_count)
        right.metric("Chunk", stats.chunk_count)
        st.metric("Maskelenen kişisel veri", stats.masked_entity_count)
        if stats.sources:
            st.caption("Kaynaklar: " + ", ".join(stats.sources))

        st.subheader("Mod")
        # A demo button may request a mode; apply it before the widget is created,
        # because Streamlit forbids writing a widget's state after it exists.
        forced = st.session_state.pop("forced_mode", None)
        if forced is not None:
            st.session_state["mode_choice"] = forced
        mode = st.radio(
            "Çalışma modu",
            options=list(AssistantMode),
            format_func=lambda m: MODE_LABELS[m],
            label_visibility="collapsed",
            key="mode_choice",
            disabled=view == VIEW_TRIAGE,
            help="Simülasyonda mod otomatik seçilir." if view == VIEW_TRIAGE else None,
        )

        with st.expander("Çalışma zamanı"):
            st.write(f"**Embedding:** {stats.embedding_backend}")
            st.write(f"**Üretim:** {assistant.llm.description}")
            st.write(f"**Eşik:** {assistant.retrieval.min_score:.2f}")

        st.caption(
            "Gizlilik: e-posta, telefon, özel link ve katılımcı listeleri indeksleme sırasında "
            "maskelenir; ham dışa aktarımlar depoya yazılmaz."
        )
    return view, mode


def render_answer(answer) -> None:
    label, kind = EVIDENCE_BADGES[answer.evidence]
    getattr(st, kind)(label)

    st.markdown(answer.text)

    if answer.evidence is not EvidenceLevel.NONE and not answer.grounded:
        st.warning(
            "Model cevabında kaynak numarası göstermedi. Aşağıdaki kaynakları kendin doğrula."
        )

    if answer.citations:
        st.subheader("Kaynaklar")
        for citation, scored in zip(answer.citations, answer.retrieved):
            authority = "Eğitmen" if citation.role.value == "instructor" else "Katılımcı"
            with st.expander(f"[{citation.index}] {citation.label} · skor {citation.score:.2f}"):
                st.markdown(f"**Yetki:** {authority} · **Eşleşme:** {scored.explanation}")
                st.text(citation.snippet)

    if answer.structured:
        with st.expander("Yapılandırılmış çıktı (JSON)"):
            st.json(answer.structured)

    st.caption(f"Üretim: {answer.generator}")


def render_triage_view(assistant: Assistant) -> None:
    st.header(VIEW_TRIAGE)
    st.warning(SIMULATION_NOTICE)
    st.caption(
        "Gruba düşmüş bir mesajı buraya yapıştır. Asistan niyeti sınıflandırır, modu kendisi "
        "seçer ve cevabın gönderilebilir mi yoksa eğitmene mi gitmesi gerektiğine karar verir."
    )

    st.write("**Örnek gelen mesajlar**")
    sample_columns = st.columns(len(INCOMING_SAMPLES))
    for column, sample in zip(sample_columns, INCOMING_SAMPLES):
        short = sample if len(sample) <= 34 else sample[:31] + "…"
        if column.button(short, use_container_width=True, help=sample, key=f"sample-{short}"):
            st.session_state["incoming_text"] = sample
            st.session_state.pop("triage", None)
            st.rerun()

    st.session_state.setdefault("incoming_text", "")
    with st.form("triage_form"):
        meta_left, meta_mid, meta_right = st.columns(3)
        channel = meta_left.selectbox(
            "Kanal",
            options=list(Channel),
            format_func=lambda c: CHANNEL_LABELS[c],
            help="Yalnızca etiket olarak saklanır; hiçbir kanala bağlanılmaz.",
        )
        group = meta_mid.text_input("Grup", value="AI Innovators - Örnek Grup")
        sender_alias = meta_right.text_input(
            "Gönderen takma adı",
            value="Katılımcı#0001",
            help="Gerçek ad yazma. Bu alan yalnızca takma ad içindir.",
        )
        text = st.text_area(
            "Gelen mesaj",
            placeholder="Örnek: GitHub repo hazır ama video çekmedim, teslim olur mu?",
            height=120,
            key="incoming_text",
        )
        analysed = st.form_submit_button("Mesajı analiz et", type="primary")

    if analysed:
        if text.strip():
            with st.spinner("Mesaj sınıflandırılıyor…"):
                st.session_state["triage"] = triage(
                    IncomingMessage(
                        text=text,
                        channel=channel,
                        group=group,
                        sender_alias=sender_alias,
                    ),
                    assistant,
                )
        else:
            st.warning("Önce bir mesaj yapıştır.")

    result = st.session_state.get("triage")
    if result is None:
        return

    label, kind = DECISION_BADGES[result.reply_decision]
    getattr(st, kind)(f"**Karar:** {label}")
    st.caption(result.reason)

    left, mid, right = st.columns(3)
    left.metric("Niyet", INTENT_LABELS[result.intent])
    mid.metric("Seçilen mod", MODE_LABELS[result.selected_mode])
    right.metric("Kanıt", EVIDENCE_BADGES[result.evidence_level][0].split(":")[-1].strip())

    if result.draft_reply:
        st.subheader("Cevap taslağı")
        st.caption("Kopyala, kontrol et, göndermeyi sen yap. Sistemde gönderim yolu yok.")
        st.code(result.draft_reply, language=None, wrap_lines=True)
    else:
        st.info("Taslak üretilmedi: kaynaklar bu mesaja cevap vermiyor.")

    if result.source_categories:
        st.caption(
            "Kaynak kategorileri: " + ", ".join(c.value for c in result.source_categories)
        )

    with st.expander("Kullanılan kaynaklar ve ham cevap"):
        render_answer(result.answer)


def render_assistant_view(assistant: Assistant, mode: AssistantMode) -> None:
    st.header(MODE_LABELS[mode])
    st.caption(
        "Cevaplar yalnızca indekslenmiş kaynaklara dayanır. Kaynaklarda karşılık yoksa asistan "
        "cevap üretmez."
    )

    st.write("**Demo soruları**")
    demo_columns = st.columns(len(DEMO_QUESTIONS))
    for column, (demo_question, demo_mode) in zip(demo_columns, DEMO_QUESTIONS):
        short = demo_question if len(demo_question) <= 42 else demo_question[:39] + "…"
        if column.button(short, use_container_width=True, help=demo_question):
            st.session_state["question_text"] = demo_question
            st.session_state["forced_mode"] = demo_mode
            st.session_state.pop("answer", None)
            st.rerun()

    st.session_state.setdefault("question_text", "")
    # A form submits the text and the click together; a bare button would lose the
    # first click to the text area's blur event.
    with st.form("ask_form"):
        question = st.text_area(
            "Soru veya durum",
            placeholder=MODE_PLACEHOLDERS[mode],
            height=120,
            key="question_text",
        )
        submitted = st.form_submit_button("Sor", type="primary")

    if submitted:
        if question.strip():
            with st.spinner("Kaynaklar taranıyor…"):
                st.session_state["answer"] = assistant.ask(question, mode=mode)
            st.session_state["asked_mode"] = mode
        else:
            st.warning("Önce bir soru veya durum yaz.")

    # Rendered outside the click branch so the answer survives later reruns.
    answer = st.session_state.get("answer")
    if answer is not None:
        if answer.mode is not st.session_state.get("asked_mode"):
            # The pipeline re-routes a confident intent instead of merely hinting at
            # it, so say which workflow actually ran rather than leaving it to guess.
            st.info(f"Bu soru **{MODE_LABELS[answer.mode]}** kapsamında değerlendirildi.")
        render_answer(answer)


def main() -> None:
    assistant = get_assistant()
    view, mode = render_sidebar(assistant)

    if assistant.is_empty:
        st.info(
            "Henüz indekslenmiş kaynak yok. Soldaki **Örnek veri** düğmesiyle başlayabilir "
            "veya kendi WhatsApp dışa aktarımını yükleyebilirsin."
        )

    if view == VIEW_TRIAGE:
        render_triage_view(assistant)
    else:
        render_assistant_view(assistant, mode)


if __name__ == "__main__":
    main()
