# Final presentation outline

Target: 8–10 minutes, roughly half of it live demo.

The through-line: *I did not build a chatbot over some documents. I found a real operational
problem in this programme, and the constraints of that problem — privacy, authority, ambiguity —
are what produced the design.*

---

## 1. The problem (1 min)

Three WhatsApp groups, three months, 2,100+ messages. The same five questions asked every week.
The answers exist, but they are two-word replies buried in a thread from July.

Two details worth stating, because they shape everything after:

- The chat history contains names, e-mail addresses, phone numbers and a roster of hundreds of
  participants. It cannot be pasted into a cloud assistant.
- Some questions were answered by participants, confidently and incorrectly.

## 2. What I built (1 min)

A local RAG assistant on Foundry Local, with four modes: instructor Q&A, submission checklist,
correction request analyzer, and technical help. Turkish answers, every one with citations.

On top of that, an incoming-message simulator: paste a message as it would arrive in a group, and
the assistant classifies it, picks the mode itself, drafts a reply, and decides whether that draft
can be used as-is or has to go to a human. The safe demo uses pasted sample messages. For an
optional Telegram pilot, the same triage path writes drafts to a local approval queue and sends
only after a human clicks approve. It never auto-sends — that is the product, not a gap.

One sentence for why local: the data is exactly the kind you must not upload, and the corpus is
small enough that a small local model with good retrieval beats a large remote one that cannot
legally see it.

## 3. Architecture (1.5 min)

One diagram, four things to say:

- Masking happens **at ingestion**, so the index never contains personal data.
- Retrieval is **hybrid** — embeddings plus lexical — because programme vocabulary is exact.
- Instructor messages are **weighted**, participant guesses damped.
- **No evidence means no model call.** The refusal path is structural.
- **Triage sits on top, not inside.** It picks the mode and judges the answer; the grounding
  contract underneath is untouched.

Do not read the module list. It is in the README.

## 4. Live demo (4 min)

Follow [`demo_script.md`](demo_script.md): grounded answer, authority weighting, checklist,
correction analyzer, refusal, then the incoming-message simulator.

Finish on the simulator, and specifically on the message that gets refused with no draft at all.
Refusal is the least common thing to demo and the most convincing; refusing inside a workflow that
could plausibly have replied is better still.

If showing the Telegram pilot as a second act, use only a pilot group and the local approval queue:
polling creates a masked pending item, the operator reviews it, and Telegram receives the message
only after the explicit approval click. Do not screenshot or commit the real approval queue.

## 5. Engineering decisions worth defending (1.5 min)

Pick two or three; do not list all of them.

- **Additive boosting was a bug.** Adding a constant to instructor chunks made every instructor
  message clear the refusal threshold for every question. Multiplying fixed it. A test for an
  out-of-scope question caught it.
- **Chunking beat scoring.** Grouping short replies with the question they answer improved results
  more than any retrieval tuning, because the most valuable instructor answers are two words long.
- **A test that scanned my own sample files found two real bugs**: chat timestamps were being read
  as phone numbers, and the "masked items" counter was measuring the wrong text.
- **Thresholds are calibrated per embedding backend**, from measurements, because a similarity
  score means different things in different embedding spaces.

## 6. Honest limitations (30 sec)

Say these before anyone asks:

- The dependency-free fallback backend refuses out-of-scope questions less reliably — 6 of 8
  versus 8 of 8 for real embeddings. Measured, documented, and the reason a real backend is
  recommended.
- No Turkish NER, so a name inside free-form prose can survive masking.
- Single-turn; no conversation memory.
- The default demo is still a simulation. Telegram is an optional local pilot, not a public webhook
  or automatic group bot.

Knowing where a system is weak is part of shipping it.

## 7. What I learned (1 min)

- Grounding is an architecture decision, not a prompt instruction.
- Privacy is easier as an ingestion-time invariant than as a display-time filter.
- Retrieval quality was mostly about the shape of the data, not the cleverness of the scoring.
- Writing the refusal path first changed every design decision that followed.

## Questions to expect

| Question | Answer |
| --- | --- |
| Why not use a cloud model? | The input data is personal. Also unnecessary: the generation step only has to summarise retrieved snippets. |
| What if Foundry Local is not running? | Retrieval still works; generation degrades to verbatim quotation, clearly labelled. Hallucination becomes impossible. |
| How do you know retrieval works? | An 18-question evaluation set: 10 in-scope answered, 8 out-of-scope refused with real embeddings. |
| Could it update the roster automatically? | It could. It should not. It drafts; a human approves. |
| Does it work on other WhatsApp exports? | Yes — both iOS and Android formats, six date variants, all tested. The instructor aliases are configuration. |
| Why is it not connected to the real WhatsApp group? | The official WhatsApp API cannot cover the existing large groups, and the unofficial route means automating WhatsApp Web against its terms of service. Neither belongs in a public repo built on other people's messages. Telegram is the optional pilot channel, with local polling and human approval. |
| So could you add auto-reply? | Technically yes, and the triage boundary is already shaped for it. I left it out on purpose: the step that puts text in front of 500 people is the one worth keeping a human on. |
