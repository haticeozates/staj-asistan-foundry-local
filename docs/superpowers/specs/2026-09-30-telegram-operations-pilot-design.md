# Telegram Operations Pilot Design

Date: 2026-09-30  
Status: Approved for planning  
Branch: `feature/telegram-operations-pilot`

## 1. Goal

Turn StajAsistan's existing local-RAG and incoming-message simulator into a
privacy-first operations pilot that can:

1. build a persistent masked index from the historical private WhatsApp corpus
   and later update exports;
2. receive new messages from an explicitly allowed Telegram group through local
   long polling;
3. classify and answer those messages with the existing triage pipeline;
4. place every answer in a durable human approval queue; and
5. send a reply only after an explicit human approval action.

This is a local pilot, not an autonomous production bot. Automatic sending,
WhatsApp Web automation, public deployment, and secrets in the repository remain
out of scope.

## 2. Current-state evidence

The current `main` branch provides:

- ingestion-time masking and pseudonymisation;
- WhatsApp export parsing;
- persistent save/load support in `NumpyVectorStore`;
- hybrid retrieval, citations and evidence levels;
- deterministic submission and correction workflows;
- `IncomingMessage -> triage -> ReplyDecision`;
- a Streamlit assistant and simulator;
- 298 passing offline tests.

The local corpus currently consists of:

- three historical private exports: 2,108 messages and 986 chunks;
- three September update ZIPs: 42 messages and 21 chunks;
- 2,150 messages before deduplication and 2,046 fingerprint-unique messages.

The September updates have no exact overlap with the historical files. There is
a 34–35 day boundary between the historical exports ending on 19 August and the
updates beginning on 22 September.

The current runtime has no private instructor alias configuration. It therefore
classifies zero real-corpus messages as instructor messages, which disables
authority weighting and measurably weakens answers. Loading the corpus without
fixing this first is not acceptable.

## 3. Chosen approach

Use two local processes that share ignored local state:

- a Telegram polling worker receives and triages messages;
- the Streamlit UI reviews, edits, approves, rejects and explicitly sends drafts.

They share a SQLite approval queue. The masked vector index remains the existing
NumPy store persisted under `data/index/`.

This is preferred over:

- a single all-in-one process, because a Streamlit rerun must not reset polling
  state or duplicate updates;
- a public webhook service, because the selected pilot is local and should not
  introduce deployment or public secret handling yet;
- ChromaDB, because the existing exact NumPy index is sufficient for roughly
  1,000 chunks and already supports persistence without another runtime
  dependency.

## 4. Private configuration

Private cohort configuration lives in `data/private/config.json`, which is
already covered by `data/private/*` in `.gitignore`.

The committed repository will contain only a schema/example with invented
values. The real file may contain:

- instructor aliases used by `InstructorIdentity`;
- stable public-safe source labels for each local input;
- allowed Telegram chat IDs;
- the persistent index location;
- the approval database location.

The Telegram token is never written to this file. It is read only from
`TELEGRAM_BOT_TOKEN` at process startup.

Loading fails closed when:

- no instructor alias is configured;
- no input paths are configured or supplied;
- a configured path escapes the expected local/private use;
- the persisted index backend does not match the active embedding backend; or
- the Telegram worker has no allowed chat ID.

## 5. Persistent corpus build

Add a corpus builder with a CLI entry point.

Inputs:

- historical `.txt`, `.md`, `.log` or `.zip` files;
- update exports;
- private instructor aliases;
- public-safe source labels;
- an output directory under `data/index/`.

Processing:

1. Read each file locally.
2. Hash the raw file in memory to identify repeat imports. The hash may be stored
   in the ignored manifest; raw bytes are never copied.
3. Parse and mask using the existing ingestion pipeline with an explicit
   `InstructorIdentity`.
4. Deduplicate only repeated input files and duplicate messages within the same
   logical source. Identical announcements posted in different groups remain
   separate provenance and are handled by retrieval MMR.
5. Build chunks and embeddings.
6. Independently scan masked messages and chunks for residual e-mail, phone and
   non-public URL patterns.
7. Refuse to persist if the scan fails or if no instructor messages are
   recognised.
8. Save `vectors.npy`, `chunks.json` and `manifest.json` atomically.

The manifest contains only:

- schema version;
- build time;
- embedding backend name and dimension;
- safe source labels and input hashes;
- source/message/chunk/instructor counts;
- masking counts;
- date range; and
- privacy scan result.

It contains no raw path, raw filename, token, instructor alias, sender name,
e-mail, phone number or private URL.

At application startup, an explicitly configured saved index is loaded instead
of starting empty. If its backend differs from the active embedding backend,
startup reports a rebuild requirement and does not query incompatible vectors.

## 6. Telegram polling boundary

Add a small Telegram transport interface:

- `get_updates(offset, timeout)`;
- `send_message(chat_id, text, reply_to_message_id)`.

The production implementation uses Telegram's official Bot API over the existing
`requests` dependency. Tests use an injected in-memory fake; the test suite never
contacts Telegram.

The poller:

1. requests text-message updates using long polling;
2. verifies the chat ID against the configured allowlist;
3. rejects bot-authored, service, non-text and unsupported updates;
4. uses Telegram `update_id` as an idempotency key;
5. masks the message body before persistence;
6. pseudonymises the sender identifier before display;
7. constructs the existing `IncomingMessage`;
8. calls the existing `triage()` function; and
9. writes the result to the approval queue.

The raw message body and real Telegram profile fields are never stored. The
queue stores the numeric chat ID and source message ID only because an approved
reply cannot be delivered without them. The database is local, ignored and
documented as sensitive operational metadata.

## 7. Approval queue and state machine

SQLite provides a durable queue shared by the poller and Streamlit.

States:

- `pending`: awaiting a human decision;
- `rejected`: explicitly declined by a human;
- `sending`: approval transaction has claimed the row;
- `sent`: Telegram accepted the message and returned a message ID;
- `failed`: send failed; the error is sanitised and retry requires another
  explicit human action.

`DO_NOT_ANSWER` results are recorded for audit but cannot be approved or sent.
`DRAFT_REPLY` and `NEEDS_HUMAN_APPROVAL` both enter `pending`: the distinction is
shown to the reviewer, but neither bypasses the human.

The approval UI permits draft editing. Clicking **Onayla ve Telegram'a gönder**:

1. atomically changes `pending` or `failed` to `sending`;
2. calls `send_message`;
3. changes the row to `sent` with the returned Telegram message ID; or
4. changes it to `failed` with a sanitised error.

Repeated clicks and concurrent reviewers cannot send the same queue item twice.
No timer, worker or background callback can transition a row to `sending`.

## 8. Streamlit changes

Keep the current Assistant and Incoming Message Simulator views. Add a third
view: **Onay Kuyruğu**.

It shows:

- masked incoming message;
- pseudonymous sender and safe group label;
- intent, selected mode, evidence and reply decision;
- source categories and citations;
- editable draft;
- pending/sent/rejected/failed state;
- reject and approve-and-send actions.

The page always states that:

- no answer is sent without an explicit click;
- tokens stay outside the repository;
- the database contains local operational metadata; and
- real-corpus screenshots must not be committed.

## 9. Security and privacy invariants

- Raw WhatsApp exports remain in ignored local storage.
- Raw Telegram text is masked before database persistence.
- Sender names/usernames are not persisted; a salted pseudonym is shown.
- Bot tokens and instructor aliases never enter git, the index manifest, logs or
  test fixtures.
- Only configured Telegram chat IDs are processed.
- `DO_NOT_ANSWER` cannot be sent.
- No outbound call occurs without an explicit approval action.
- Outbound text is limited to the human-reviewed draft.
- Network errors are sanitised before storage and display.
- The repository continues to contain sample data only.

## 10. Testing

All implementation follows red-green-refactor.

Corpus tests:

- missing instructor aliases fail closed;
- repeated file hashes do not duplicate the index;
- same-source duplicate messages are removed;
- cross-source provenance is preserved;
- manifest contains no raw paths, aliases or secrets;
- backend mismatch refuses load;
- failed privacy scan writes no index;
- valid saved index round-trips.

Telegram tests:

- unallowed chat is ignored;
- non-text and bot messages are ignored;
- duplicate `update_id` creates one queue row;
- raw sender identity is not persisted;
- message text is masked before persistence;
- poll offset advances correctly;
- token is read from the environment only.

Approval tests:

- pending rows are never sent automatically;
- `DO_NOT_ANSWER` cannot be approved;
- reject sends nothing;
- one explicit approval sends exactly once;
- edited draft is the exact outbound text;
- concurrent or repeated approval sends once;
- transport failure becomes `failed`;
- retry requires another explicit approval.

UI tests:

- queue renders masked metadata and policy;
- approval controls are unavailable for `DO_NOT_ANSWER`;
- explicit approval invokes the approval service;
- existing assistant and simulator regressions remain green.

Private acceptance (not committed):

- build the six-file corpus locally;
- require at least one instructor message and a clean privacy scan;
- run labelled in-scope and out-of-scope questions;
- verify correction and sensitive topics require human approval;
- verify no real identifier appears in stored chunks or queue rows.

## 11. Documentation and delivery

Update:

- README setup/run instructions;
- architecture and privacy docs;
- an ignored private-config example generator;
- Telegram BotFather privacy-mode and allowed-chat setup;
- local corpus build/run commands;
- backup and deletion instructions for local index and queue;
- honest limitations and WhatsApp Groups API constraints.

No real token, chat ID, alias, export filename, private path or real message is
included in documentation, tests, screenshots, commits or final reports.

## 12. Acceptance criteria

The pilot is complete when:

1. the six-file real corpus builds into a persistent masked index with recognised
   instructor authority and zero structured-identifier leaks;
2. restarting the app reloads the index without reading raw exports again;
3. an allowed Telegram group message creates exactly one queue item;
4. no queue item sends without an explicit human action;
5. an edited approved draft is sent exactly once as a reply to the source
   message;
6. rejected and `DO_NOT_ANSWER` items never send;
7. secrets and private data are absent from git;
8. all offline tests pass; and
9. the final private smoke report contains aggregate metrics only.
