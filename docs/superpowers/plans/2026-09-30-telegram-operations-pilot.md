# Telegram Operations Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent masked private-corpus index and a local Telegram polling pilot whose replies can be sent only by an explicit human approval action.

**Architecture:** A corpus builder persists the existing NumPy vector store plus a privacy-safe manifest. A Telegram poller and Streamlit approval view share an ignored SQLite queue; the poller only creates queue rows, while an approval service is the sole outbound boundary.

**Tech Stack:** Python 3.11+, NumPy, SQLite (`sqlite3`), `requests`, Streamlit, pytest.

## Global Constraints

- Raw exports, Telegram tokens, instructor aliases, chat IDs and raw sender identities never enter git.
- `data/private/`, `data/index/` and local queue databases remain ignored.
- Real corpus text is never printed in test output or final reports.
- Hashing embeddings are test-only; a private build requires `foundry` or `sentence-transformers` unless `--allow-hashing` is explicitly supplied for diagnostics.
- `DO_NOT_ANSWER` can never be approved or sent.
- No outbound Telegram call occurs without an explicit human approval action.
- Tests inject a fake Telegram transport and never access the network.

---

### Task 1: Persistent private corpus

**Files:**
- Create: `src/staj_asistan/private_config.py`
- Create: `src/staj_asistan/corpus.py`
- Create: `tests/test_corpus.py`
- Modify: `src/staj_asistan/pipeline.py`
- Modify: `src/staj_asistan/cli.py`

**Interfaces:**
- Produces: `PrivateConfig.load(path: Path) -> PrivateConfig`
- Produces: `build_private_index(config, inputs, output_dir, embedding_backend, allow_hashing=False) -> CorpusManifest`
- Produces: `load_private_assistant(index_dir, llm=None) -> Assistant`
- Produces: CLI `staj-asistan corpus build --config ... --output ... INPUT...`

- [ ] **Step 1: Write failing config and corpus tests**

```python
def test_private_config_requires_instructor_alias(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"instructor_aliases": []}')
    with pytest.raises(ValueError, match="instructor"):
        PrivateConfig.load(path)

def test_manifest_contains_no_raw_path_alias_or_secret(tmp_path, hashing_backend):
    manifest = build_fixture_index(tmp_path, hashing_backend)
    blob = (tmp_path / "index" / "manifest.json").read_text()
    assert "/Users/" not in blob
    assert "Örnek Eğitmen" not in blob
    assert "token" not in blob.lower()
    assert manifest.privacy_scan_clean is True

def test_saved_index_round_trips(tmp_path, hashing_backend):
    manifest = build_fixture_index(tmp_path, hashing_backend)
    assistant = load_private_assistant(tmp_path / "index", llm=ExtractiveClient())
    assert assistant.stats.chunk_count == manifest.chunk_count
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -o addopts= -q tests/test_corpus.py --tb=short`  
Expected: collection error because `staj_asistan.corpus` and `private_config` do not exist.

- [ ] **Step 3: Implement private configuration**

```python
@dataclass(frozen=True)
class PrivateConfig:
    instructor_aliases: tuple[str, ...]
    allowed_telegram_chat_ids: frozenset[int] = frozenset()
    source_labels: dict[str, str] = field(default_factory=dict)
    index_dir: Path = Path("data/index/private")
    queue_db: Path = Path("data/index/telegram-queue.sqlite3")

    @classmethod
    def load(cls, path: Path) -> "PrivateConfig": ...
```

Reject empty aliases, non-integer chat IDs and paths outside `data/index`. Never expose
configuration contents in exception messages.

- [ ] **Step 4: Implement corpus manifest and builder**

```python
@dataclass(frozen=True)
class CorpusManifest:
    schema_version: int
    embedding_backend: str
    embedding_dimension: int
    source_count: int
    message_count: int
    instructor_message_count: int
    chunk_count: int
    masked_entity_count: int
    source_labels: tuple[str, ...]
    input_hashes: tuple[str, ...]
    privacy_scan_clean: bool
```

Read each input once, deduplicate identical file SHA-256 hashes, ingest with an explicit
`InstructorIdentity`, scan masked chunks, require at least one instructor chunk, save into a
temporary sibling directory, then atomically rename it into place.

- [ ] **Step 5: Add assistant index loading**

Add `Assistant.from_saved_index(index_dir, llm=None)` or the equivalent corpus helper. Compare
the manifest backend name and stored vector dimension with the resolved embedding backend before
constructing the retriever.

- [ ] **Step 6: Add corpus CLI**

The command prints aggregate counts only:

```text
Index built: 6 sources, 2150 messages, 1007 chunks
Instructor messages: <count>
Privacy scan: clean
```

- [ ] **Step 7: Run GREEN**

Run: `.venv/bin/pytest -o addopts= -q tests/test_corpus.py --tb=short`  
Expected: all corpus tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/staj_asistan/private_config.py src/staj_asistan/corpus.py \
  src/staj_asistan/pipeline.py src/staj_asistan/cli.py tests/test_corpus.py
git commit -m "feat: build persistent masked private corpus"
```

---

### Task 2: Durable approval queue

**Files:**
- Create: `src/staj_asistan/approval_queue.py`
- Create: `tests/test_approval_queue.py`

**Interfaces:**
- Produces: `QueueStatus`
- Produces: `ApprovalItem`
- Produces: `ApprovalQueue.enqueue(...)`, `list_items(...)`, `get(...)`,
  `reject(...)`, `claim_for_send(...)`, `mark_sent(...)`, `mark_failed(...)`

- [ ] **Step 1: Write failing queue tests**

```python
def test_update_id_is_idempotent(queue, triage_result):
    first = queue.enqueue(update_id=7, triage_result=triage_result, ...)
    second = queue.enqueue(update_id=7, triage_result=triage_result, ...)
    assert first.id == second.id
    assert len(queue.list_items()) == 1

def test_do_not_answer_cannot_be_claimed(queue, refusal):
    item = queue.enqueue(update_id=8, triage_result=refusal, ...)
    with pytest.raises(InvalidTransition):
        queue.claim_for_send(item.id, edited_draft="anything")

def test_claim_is_atomic(queue, draft):
    item = queue.enqueue(update_id=9, triage_result=draft, ...)
    claimed = queue.claim_for_send(item.id, edited_draft="Reviewed")
    assert claimed.status is QueueStatus.SENDING
    with pytest.raises(InvalidTransition):
        queue.claim_for_send(item.id, edited_draft="duplicate")
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -o addopts= -q tests/test_approval_queue.py --tb=short`  
Expected: missing module.

- [ ] **Step 3: Implement SQLite schema and transitions**

Use WAL mode, `BEGIN IMMEDIATE` for claims, a unique `update_id`, and parameterised SQL.
Persist masked incoming text, pseudonymous sender, safe group label, chat/message IDs, triage
metadata, draft, status and sanitised error. Do not persist raw sender profile fields.

- [ ] **Step 4: Run GREEN and commit**

```bash
.venv/bin/pytest -o addopts= -q tests/test_approval_queue.py --tb=short
git add src/staj_asistan/approval_queue.py tests/test_approval_queue.py
git commit -m "feat: add durable human approval queue"
```

---

### Task 3: Telegram transport, poller and approval service

**Files:**
- Create: `src/staj_asistan/telegram.py`
- Create: `src/staj_asistan/telegram_worker.py`
- Create: `tests/test_telegram.py`

**Interfaces:**
- Produces: `TelegramTransport` protocol
- Produces: `TelegramHTTPTransport.from_env()`
- Produces: `TelegramPoller.process_update(update: dict) -> int | None`
- Produces: `ApprovalService.approve_and_send(item_id, edited_draft) -> ApprovalItem`

- [ ] **Step 1: Write failing transport boundary tests**

```python
def test_unallowed_chat_is_ignored(poller):
    assert poller.process_update(text_update(chat_id=999)) is None
    assert poller.queue.list_items() == []

def test_message_is_masked_and_sender_is_pseudonymous(poller):
    poller.process_update(text_update(text="mail me at real@private.test"))
    item = poller.queue.list_items()[0]
    assert "real@private.test" not in item.incoming_text
    assert item.sender_alias.startswith("Katılımcı#")

def test_approval_sends_exactly_once(service, queue, fake_transport, draft):
    item = queue.enqueue(update_id=10, triage_result=draft, ...)
    service.approve_and_send(item.id, "Human edited draft")
    assert fake_transport.sent == [(item.chat_id, "Human edited draft", item.message_id)]
    with pytest.raises(InvalidTransition):
        service.approve_and_send(item.id, "again")
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -o addopts= -q tests/test_telegram.py --tb=short`  
Expected: missing Telegram modules.

- [ ] **Step 3: Implement HTTP transport**

Use `requests.Session`, bounded connect/read timeouts, `getUpdates` and `sendMessage`. Read
`TELEGRAM_BOT_TOKEN` once in `from_env`; never include URLs containing the token in errors.

- [ ] **Step 4: Implement poller**

Accept only allowlisted group/supergroup text messages from non-bot senders. Mask before queue
storage, create `IncomingMessage(channel=Channel.TELEGRAM)`, and enqueue triage output. Advance
offset after every observed update, including ignored ones.

- [ ] **Step 5: Implement approval service**

Claim row atomically, send reviewed text, mark sent or failed. Sanitise network exceptions.

- [ ] **Step 6: Run GREEN and commit**

```bash
.venv/bin/pytest -o addopts= -q tests/test_telegram.py --tb=short
git add src/staj_asistan/telegram.py src/staj_asistan/telegram_worker.py tests/test_telegram.py
git commit -m "feat: add Telegram polling with approval-gated replies"
```

---

### Task 4: Streamlit approval view

**Files:**
- Create: `src/staj_asistan/approval_ui.py`
- Modify: `app.py`
- Modify: `tests/test_triage.py`

**Interfaces:**
- Consumes: `ApprovalQueue`, `ApprovalService`
- Produces: third Streamlit view `Onay Kuyruğu`

- [ ] **Step 1: Add failing headless UI tests**

```python
def test_queue_view_states_no_automatic_send(app):
    app.sidebar.radio[0].set_value("Onay Kuyruğu").run()
    assert any("açık onay" in warning.value for warning in app.warning)

def test_refusal_has_no_send_control(app_with_refusal):
    app_with_refusal.sidebar.radio[0].set_value("Onay Kuyruğu").run()
    assert not any("Onayla ve Telegram'a gönder" == b.label for b in app_with_refusal.button)
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -o addopts= -q tests/test_triage.py -k queue_view --tb=short`  
Expected: queue view option missing.

- [ ] **Step 3: Implement focused approval UI module**

Keep queue rendering outside `app.py`. Show masked input, triage metadata, citations, editable
draft and state. Render approve only for pending/failed non-refusals; render reject only for
pending items.

- [ ] **Step 4: Wire dependencies in app**

Load the saved index and queue only when local paths are configured. The sample-data demo remains
the safe default when no private config is present.

- [ ] **Step 5: Run GREEN and commit**

```bash
.venv/bin/pytest -o addopts= -q tests/test_triage.py --tb=short
git add app.py src/staj_asistan/approval_ui.py tests/test_triage.py
git commit -m "feat: add Telegram approval queue view"
```

---

### Task 5: Private acceptance, documentation and final verification

**Files:**
- Create: `docs/telegram_pilot.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/privacy.md`
- Modify: `.env.example`

- [ ] **Step 1: Update documentation**

Document private config creation, semantic backend requirement, corpus build, BotFather
`/setprivacy`, allowed chat IDs, polling worker start, approval/reject flow, index/queue deletion,
and the official WhatsApp Groups API constraints.

- [ ] **Step 2: Run full offline suite**

Run: `.venv/bin/pytest -o addopts= -q --tb=line`  
Expected: all tests pass; no network access.

- [ ] **Step 3: Build the real private index**

Use an ignored `data/private/config.json` and the six local inputs. Do not print aliases, raw
paths, message text or citations. Require:

```text
instructor_message_count > 0
privacy_scan_clean = true
message_count >= 2150
chunk_count > 0
```

- [ ] **Step 4: Run private aggregate smoke tests**

Test submission, correction, channel, certificate and out-of-scope queries. Report only intent,
mode, evidence, decision and citation count.

- [ ] **Step 5: Privacy and git audit**

Run repository scans for e-mail, phone, local path, tokens, chat IDs, instructor aliases and raw
export names. Verify `git status --short` contains only intended source/docs changes and no local
index/database/config.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/architecture.md docs/privacy.md docs/telegram_pilot.md .env.example
git commit -m "docs: document private Telegram operations pilot"
```

- [ ] **Step 7: Final handoff**

Report test result, private aggregate build metrics, privacy scan, `git status --short`,
`git log --oneline -5`, local run commands and whether the branch was pushed. Do not print any
secret or private identifier.
