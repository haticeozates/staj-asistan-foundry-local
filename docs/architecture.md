# Architecture

How a question becomes a grounded answer, and why each stage is shaped the way it is.

## Pipeline

```text
raw export ─▶ ingestion ─▶ parsing ─▶ masking ─▶ chunking ─▶ embedding ─▶ index
                                                                            │
question ─▶ retrieval (dense + lexical, weighted) ─▶ threshold ─┬─ below ─▶ refusal
                                                                └─ above ─▶ prompt ─▶ model ─▶ answer + citations

incoming message ─▶ triage ─▶ intent ─▶ mode ─▶ (the pipeline above) ─▶ reply decision
                                                                         │
Telegram getUpdates ─▶ mask ─▶ triage ─▶ SQLite queue ─▶ human approve ─▶ sendMessage
```

The refusal branch is the important one: when nothing clears the threshold, the language model is
never called at all. A model that is not invoked cannot hallucinate.

Triage sits on top rather than inside: it chooses how to ask, and judges what may be done with the
answer, but the grounding contract below it is unchanged.

## 1. Ingestion (`ingestion.py`)

Accepts `.txt`, `.md`, `.log` and `.zip`. Inside an archive only text entries are read — the image
and audio files that WhatsApp exports alongside the chat are never opened, which removes the
largest privacy risk in a single decision.

Each file is classified as a chat export or a plain document by a ratio heuristic over the first
lines, so a two-message export is still recognised as a chat rather than falling through to the
document path.

## 2. Parsing (`whatsapp_parser.py`)

Handles both iOS (`[17.06.2026 16:13:12] Sender: text`) and Android (`17.06.2026, 16:13 - Sender: text`)
formats, 12- and 24-hour clocks, and two-digit years.

Three things here matter more than they look:

- **Multiline messages.** The instructor's most substantial announcements span several lines. A
  line-per-message parser would shred exactly the content that most needs to stay whole, so any
  line that does not start with a timestamp is appended to the message above it.
- **Bidi control characters.** WhatsApp wraps phone numbers in Unicode direction marks. They are
  invisible and they break every regex, so they are stripped before anything else runs.
- **System events.** Join/leave notices, "messages are end-to-end encrypted", deleted messages and
  attachment stubs are dropped. In the real corpus this removed several hundred lines that would
  otherwise have become retrievable, meaningless chunks.

Instructor identification is alias-based, covering the name with and without the organisation
suffix, ASCII-folded spelling variants, and the exported phone-number form.

## 3. Masking (`privacy.py`)

Runs on the message body before it becomes a `Message`, so nothing downstream ever holds
unmasked text. Details in [`privacy.md`](privacy.md).

## 4. Chunking (`chunking.py`)

Two strategies, because the corpus has two very different shapes.

- **Long announcements** are split on paragraph boundaries with overlap, so a rule that spans a
  paragraph break is not cut in half.
- **Short messages** are grouped into conversation windows: consecutive messages within a time gap
  and character budget become one chunk.

The second one carries most of the retrieval quality. The instructor's answers are frequently two
words long — *"Sertifika tek."* — and are worthless in isolation; grouped with the question they
answer, they become the best chunk in the index. A window is also cut when authority changes, so a
participant's guess is never merged into an instructor chunk.

Every chunk keeps its source, time range, message count and instructor ratio, which is what makes
citations and authority weighting possible later.

## 5. Embedding (`embeddings.py`)

A three-step resolution chain, always reported in the UI so the active backend is never a mystery:

1. `foundry` — Foundry Local `/v1/embeddings`
2. `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2`, the recommended setup
3. `hashing` — a deterministic character n-gram vectoriser with no dependencies

The hashing backend exists so the test suite runs in about a second with no model download and no
network, and so a first launch is never a five-minute wait. It is a fallback, not a peer: see the
measurements below.

## 6. Vector store (`vector_store.py`)

Default is exact cosine similarity over a NumPy matrix. At ~1,000 chunks an approximate index would
add a dependency and a failure mode to solve a problem this corpus does not have. ChromaDB is
supported behind the same interface for anyone who needs persistence at a larger scale.

## 7. Retrieval (`retriever.py`)

Score for a chunk against a query:

```text
relevance = 0.65 · cosine(dense) + 0.35 · normalised BM25
score     = relevance · authority · recency
```

**Why lexical scoring is kept.** Program vocabulary is narrow and exact: *sertifika*, *teslim*,
*video*, *batch*. Dense similarity alone retrieves topically-near chunks that miss the specific
word the user asked about.

**Authority weighting is multiplicative, and that was a bug fix.** The first version added a
constant to instructor chunks. That silently defeated the refusal path — any instructor message
scored above the threshold for any question, including questions the corpus does not answer.
Multiplying scales what is already relevant instead of promoting what is not. Participant chunks
are damped for the mirror-image reason: the corpus contains confident wrong answers from
participants, and they must not outrank the instructor's correction. Curated documents are neither
boosted nor damped.

**Recency** decays with a half-life of about 45 days. Rules changed mid-programme, so a July
announcement should lose to an August one on the same topic.

**MMR de-duplication.** The instructor reposted the same submission rules roughly every two weeks.
Without diversity filtering, all six retrieved slots go to six copies of one announcement; with it,
the context window holds six different facts.

**The refusal gate** has two parts:

1. The best score must clear a per-backend minimum.
2. The query must share at least one *discriminative* term with the corpus — a term that does not
   appear in more than a set fraction of chunks. Stop-word-ish overlap ("nasıl", "için") does not
   count as evidence that the corpus has anything to say.

Thresholds are calibrated per backend, because a similarity number means different things to
different embedding spaces. Sharing one cut-off across backends would make it wrong for at least
one of them.

Turkish is handled with fixed-length stem truncation. It is a crude morphological analyser but a
strong baseline: it collapses *teslim / teslimi / tesliminde* without a dependency.

### Measured behaviour

Evaluation set: 12 questions the corpus answers, 9 questions it does not.

| Backend | Answers in-scope (of 12) | Correctly refuses out-of-scope (of 9) |
| --- | --- | --- |
| `sentence-transformers` | 12 | 9 |
| `hashing` | 12 | 4 |

The hashing backend accepts questions like *"Mars'a ne zaman insan gönderilecek?"* because a
bag-of-words vector cannot tell that sharing the stems *zaman* and *gönder* is not topical overlap.
This is the documented reason a real embedding backend is recommended for anything but tests.

## 8. Generation (`generation.py`, `prompts.py`)

Foundry Local is reached through its OpenAI-compatible endpoint, discovered via the SDK when
available and falling back to a plain HTTP client.

The prompt carries numbered sources and a grounding contract: answer only from the sources, cite
with `[n]`, and say the sources are unclear rather than filling the gap. Citation markers pointing
at sources that were not supplied are stripped from the output, so a fabricated reference cannot
reach the screen.

If no runtime is available, `ExtractiveClient` takes over and quotes retrieved sentences verbatim.
Fluency drops and the answer is clearly labelled as quotation, but the guarantee gets stronger
rather than weaker.

## 9. Workflows (`workflows.py`)

Checklist and correction analysis are deterministic rules, not model output, because a checklist
that changes between runs is not a checklist.

Both operate on clauses and handle Turkish negation explicitly: *"video çektim"* and *"video
çekmedim"* differ by three characters and mean opposite things, and a keyword matcher gets that
backwards every time.

The correction analyzer deliberately produces a draft and never an action. It also reports what is
*missing* — a roster correction without a full name and e-mail address cannot be matched to a row,
which is the actual failure mode this mode exists to prevent.

## 10. Pipeline (`pipeline.py`)

Wires the stages together and owns the grounding contract: retrieve, decide the evidence level,
call the model only if there is evidence, attach citations, verify the markers.

Retrieval parameters are tuned per mode — the checklist mode wants breadth across submission rules,
the Q&A mode wants precision on a single question.

### Routing, not hinting

Instructor Q&A is the default, not a choice, so a confident intent overrides it: a correction
request typed into the default view runs through the correction workflow, and the UI states which
workflow answered. An explicitly selected mode is never overridden — if somebody opened Technical
Help deliberately, second-guessing them is worse than answering what they asked.

The earlier behaviour was a caption suggesting a better mode while answering in the wrong one.
That is the worst of both: the user is told the answer is unsuitable and given it anyway.

### The topic floor

Refusal used to depend entirely on the retrieval threshold, which left a hole. The deterministic
modes build their cards from rules rather than from retrieval, so *"İstanbul'da hava nasıl?"* in
checklist mode produced a full submission checklist with nothing behind it.

The fix is a mode-independent gate applied before dispatch: if the question shares no programme
vocabulary *and* the evidence is not high, refuse. Both halves are load-bearing.

The vocabulary check alone is too blunt — measured on sample data, *"Ne zaman başlıyoruz?"*,
*"İki tane yapsam ne olur?"*, *"Başka bir dil kullanabilir miyim?"* and *"Nereye göndereceğim?"*
contain no programme noun yet all four retrieve correct instructor answers at high evidence.
Retrieval must keep the final say when it is confident.

The evidence check alone was the original hole. Together they took the offline backend's
out-of-scope refusal rate from 3/8 to 6/8 without silencing a single in-scope question.

A refusal returns the refusal text and nothing else: no citations, no retrieved chunks, no
structured card. A refusal that still renders sources reads as a hedged answer.

## 11. Triage (`triage.py`)

Everything above assumes a person opened the tool and chose a mode. Triage models the case that
actually occurs: a message lands in a group and something has to decide what to do with it.

It is a thin layer on purpose. Intent classification reuses the routing heuristics that already
exist — `is_submission_channel_query`, `is_submission_requirement_query`, `suggest_mode` and the
retrieval-side `infer_query_topic` — instead of growing a second, competing set of keyword lists
that would drift out of sync with the first. Order matters more than the individual checks: the
roster vocabulary overlaps nearly everything, so *"listede projem yanlış"* has to be recognised as
a correction before the word "liste" pulls it into the roster lane.

Falling back to `infer_query_topic` also closes a real gap. The mode hints do not list
*"çalışmazsa"*, so *"Foundry Local çalışmazsa ne kontrol etmeliyim?"* used to route to general Q&A;
the retrieval vocabulary does list it, and triage now picks technical help.

### The reply policy

Three outcomes, applied in this precedence:

| Order | Condition | Decision |
| --- | --- | --- |
| 1 | No evidence at all | Do not answer |
| 2 | Correction request | Needs human approval |
| 3 | Sensitive topic — certificate, deadline, official document, personal record | Needs human approval |
| 4 | The message itself carries personal data | Needs human approval |
| 5 | Low evidence, answer from a deterministic rule card | Needs human approval |
| 6 | Low evidence, everything else | Do not answer |
| 7 | Medium or high evidence | Draft reply |

Rules 5 and 6 are the interesting split. Treating low evidence as an automatic refusal throws away
correct answers: *"WhatsApp'tan yazmam yeterli mi?"* is answered by a deterministic rule card, and
the low score means the retrieved snippets corroborate it thinly, not that the answer is doubtful.
Escalating instead of refusing keeps the answer available to a human who can confirm it in seconds.

Rules 2 through 4 fire regardless of how strong the evidence is. A confidently-worded certificate
answer is more dangerous than an uncertain one, not less, because the cost of being wrong lands on
a participant who cannot undo it.

### No auto-send, one gated outbound path

Triage still has no fourth decision. `as_dict()` always includes `auto_send: False`. The channel
on an `IncomingMessage` is metadata: the same message triaged as WhatsApp, Telegram or manual
input yields a byte-identical decision and draft.

Sending lives outside triage, in three modules that did not exist on the demo path:

- `telegram.py` talks to the Bot API. The token comes only from `TELEGRAM_BOT_TOKEN`. Exceptions
  and logs are redacted. A failed `sendMessage` is classified as rejected (safe to retry) or
  uncertain (must not retry).
- `telegram_worker.py` splits inbound and outbound. `TelegramPoller` long-polls, allowlists,
  masks, triages and enqueues. It never calls `sendMessage`. `ApprovalService` is the only send
  path: atomic claim, one send, then `sent` / `failed` / locked-`sending`.
- `approval_ui.py` renders the queue. A transport is constructed only inside the send-button
  callback, after a confirmation checkbox. `DO_NOT_ANSWER` cannot be claimed.

The sample-data demo loads none of this. Streamlit opens the private index and queue only when
`STAJ_ASISTAN_PRIVATE_CONFIG` points at a valid local file.

## 12. Persistent private corpus (`corpus.py`, `private_config.py`)

The demo index is rebuilt from `data/samples/` in memory. The operations index is a masked NumPy
store under `data/index/`, written atomically with a manifest (backend, dimension, counts,
privacy-scan flag). Input files are SHA-256 deduplicated. Instructor aliases and Telegram chat
IDs stay in gitignored `data/private/config.json` and never enter the manifest. Hashing embeddings
require an explicit `--allow-hashing` flag so a private build cannot silently use the test
backend.

## 13. Two-process operations runtime

The poller and Streamlit share SQLite (WAL) and must not share a process: a Streamlit rerun would
reset the Telegram offset. Operator steps, BotFather `/setprivacy`, and WhatsApp API limits are
in [`telegram_pilot.md`](telegram_pilot.md).
