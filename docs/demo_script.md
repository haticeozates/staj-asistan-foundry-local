# Demo script

Roughly six minutes. Every question below is a question that was actually asked, repeatedly, in
the programme's WhatsApp groups.

## Before you start

```bash
foundry service start
foundry model run phi-4-mini

export STAJ_ASISTAN_EMBEDDING_BACKEND=sentence-transformers
streamlit run app.py
```

Then press **Temizle**, and only after that **Örnek veri**. Do this even on a fresh start, and do
it before you hit record.

This order is not fussiness. The index is cumulative: if you tested against a real export earlier
in the session, its chunks are still in there and will surface as citations mid-demo. **Temizle**
empties the store, **Örnek veri** reloads only `data/samples/`, and the sidebar counters let you
confirm it — three files, and a chunk count that matches sample data rather than a real corpus.

Check that the sidebar shows the embedding backend and the Foundry Local model — the audience
should see that both are local before any answer appears.

Demo on the anonymised sample data. The real corpus is where the system was validated, but its
snippets are other people's messages and do not belong on a projector, in a recording, or in a
screenshot. Real exports are for local, private testing only.

If Foundry Local is not running, everything still works: answers switch to quotation mode and are
labelled as such. That is worth showing on purpose if you have a spare thirty seconds.

## 1. "Final tesliminde ne gerekiyor?" — grounded answer

*Submission Checklist mode.* You can leave the sidebar on Instructor Q&A and it still lands there:
submission-rule questions are re-routed automatically, because in practice nobody picks the mode
before asking. Worth mentioning in one sentence — it is the same routing the simulator uses later.

Expect a short Turkish answer covering the code link, the two-minute video and sending both by
e-mail, with numbered citations underneath.

**Point at the citations.** Each one shows the instructor's own message, its date, and why it was
retrieved. This is the difference between an answer and a claim.

## 2. "Liste güncel değilse çalışmaya devam etmeli miyim?" — authority weighting

*Instructor Q&A mode*, and this one genuinely stays there.

Expect: keep working, the roster updates with a delay.

**The point of this question** is that participants answered it too, and some of them answered it
wrongly. The instructor's reply is short — *"Liste gecikmeli güncelleniyor, siz çalışmaya devam
edin."* — and would lose a pure similarity contest against longer, chattier messages. It wins here
because instructor messages scale up and participant guesses scale down.

Worth saying out loud: this weighting is multiplicative, not additive, because an additive version
broke refusal. That story is in `docs/architecture.md` if it comes up.

## 3. "GitHub repo hazır ama video çekmedim, teslim için eksiğim var mı?" — checklist

*Submission Checklist mode.*

Expect three sections: what is ready (code link, README), what is missing (the video, and sending
the links by e-mail), and concrete next steps.

**Two things to highlight.** First, it read *"video çekmedim"* as an absence, not as the presence
of the word "video" — Turkish negation is handled explicitly. Second, the checklist is rule-based,
so it produces the same answer every time; the model may only rephrase it.

## 4. "Listede projem yanlış görünüyor, Foundry Local olarak güncellenmesini istiyorum. Bu mesaj yeterli mi?" — correction analyzer

*Correction Request Analyzer mode* — and you can leave the sidebar on Instructor Q&A to prove it.
The banner above the answer will read *"Bu soru Düzeltme İsteği Analizi kapsamında değerlendirildi"*.
The assistant routes the message rather than suggesting you re-ask it somewhere else.

Expect a header stating the three things a participant actually needs to hear — this is a
correction request, the assistant cannot apply it, only the instructor can — followed by
structured JSON:

```json
{
  "intent": "list_correction",
  "detected_fields": { "project": "Building Your First Local RAG Application with Foundry Local" },
  "missing_required_info": ["full_name", "email"],
  "confidence": "low",
  "auto_apply": false
}
```

plus a ready-to-send draft e-mail.

**This is the strongest slide.** The answer to "is this message enough?" is *no*, and the system
says so with a reason: without a full name and e-mail address the instructor cannot identify which
row to fix. That is the real failure mode in the programme — corrections that cannot be matched to
a person.

Say the quiet part explicitly: `auto_apply` is always false. The assistant drafts, a human decides.
An agent that edits a roster on the strength of a chat message is a liability, not a feature.

## 5. "İstanbul'da hava nasıl?" — refusal

Ask something the corpus cannot answer.

Expect two sentences and nothing else: *"Bu soru yüklü kaynaklarda yer almıyor"*, followed by the
topics it does cover. **Point at what is missing** — no snippet, no citation panel, no "kısa
cevap" block, no half-filled checklist. A refusal that still shows sources reads as a hedged
answer, and a hedged answer about programme rules is worse than silence.

Switch the mode to Teslim Kontrol Listesi and ask it again. It refuses identically; the
deterministic workflows do not get to build a card out of an off-topic question.

The model was never called. Retrieval found nothing above threshold, so generation was skipped
entirely — the guarantee is structural, not a polite instruction in a prompt. Most demos avoid this
question; it is the one worth showing.

## 6. Incoming message simulation — the operations story

Switch the sidebar **Görünüm** to *Gelen Mesaj Simülasyonu*. Read the standing warning aloud once:
the system never sends anything, it only drafts for a human.

Run three of the sample buttons and let the decision column tell the story:

| Message | Decision | Why it matters |
| --- | --- | --- |
| *GitHub repo hazır ama video çekmedim, teslim olur mu?* | Draft reply | Routine, well-covered question. The draft is copyable as-is. |
| *Listede projem yanlış görünüyor…* | Needs human approval | Strong evidence, still escalated. Only the instructor can edit a roster. |
| *İstanbul'da hava nasıl?* | Do not answer | No evidence, and therefore no draft at all — not even a hedged one. |

**Two points to land.** First, nobody chose a mode: the assistant read the message, picked
Submission Checklist, Correction Analyzer and Instructor Q&A by itself, and showed which one it
picked. Second, and this is the close: there is no send button, and there is no sending code behind
the screen either. The escalation rules fire *on top of* strong evidence, because a confident wrong
answer about somebody's certificate is worse than an uncertain one, not better.

If asked why this is not connected to a real group: the official WhatsApp Business API needs a
reviewed business account, and the unofficial route means automating WhatsApp Web against its terms
of service. Neither belongs in a public repository built on other people's messages.

## If something goes wrong

| Symptom | Cause | Fix |
| --- | --- | --- |
| Every answer starts "Yerel dil modeli çalışmadığı için…" | Foundry Local is not reachable | `foundry service start`, then reload |
| Answers are oddly off-topic | The `hashing` fallback is active | Set the embedding backend and restart |
| Sidebar shows 0 chunks | Nothing indexed yet | Press **Örnek veri** |
| A citation quotes a message you recognise | A real export is still in the index | **Temizle**, then **Örnek veri**, and re-record |
