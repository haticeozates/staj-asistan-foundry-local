# Telegram operations pilot

A local, human-in-the-loop path from an allowlisted Telegram group to a cited
draft, and from that draft to the group **only** after an explicit approval
click. Automatic sending is not implemented.

The sample-data Streamlit demo is unchanged: without a private config it still
answers from anonymised samples and has no Telegram connection.

## What this is not

- Not WhatsApp Web automation, and not an unofficial WhatsApp client.
- Not a public webhook service and not a cloud bot.
- Not a way to join an existing large WhatsApp group. WhatsApp's official
  Groups API is currently limited to API-created invite groups with an
  eight-participant cap and Cloud API on-behalf-of-a-user access. That cannot
  cover the internship groups, which is why this pilot uses Telegram.

## Prerequisites

1. Python 3.11+, this repository, and a semantic embedding backend
   (`sentence-transformers` or Foundry Local embeddings). The hashing backend is
   for tests only; private index builds refuse it unless you pass
   `--allow-hashing`.
2. A Telegram bot from BotFather. In BotFather run `/setprivacy` and disable
   privacy mode so the bot can see group messages that are not commands.
3. The bot added to the programme group (or a dedicated pilot group).
4. The numeric chat ID of that group. Telegram group IDs are negative integers.
   Store them only in the ignored private config, never in git.
5. `TELEGRAM_BOT_TOKEN` in the local environment, never in a committed file.

## Private configuration

Create `data/private/config.json`. That path is already gitignored. Do not put
real aliases, chat IDs or tokens in this repository.

```json
{
  "instructor_aliases": ["Ad Soyad", "Ad Soyad Kurum"],
  "allowed_telegram_chat_ids": [-1001234567890],
  "source_labels": {
    "export-one.txt": "Grup A"
  },
  "index_dir": "data/index/private",
  "queue_db": "data/index/telegram-queue.sqlite3"
}
```

Rules the loader enforces:

- At least one instructor alias is required. Without it the real corpus scores
  as zero instructor messages and authority weighting collapses.
- Chat IDs must be integers. An empty list is valid for index builds, but the
  poller will refuse to start until at least one ID is present.
- `index_dir` and `queue_db` must be relative paths under `data/index/`.
- Source labels are optional. If omitted, inputs become `Kaynak 1`, `Kaynak 2`,
  … Labels must not contain personal data or an instructor alias.
- Instructor aliases are stripped from message bodies during the build and
  replaced with the generic role label `Eğitmen`, so a mention of the person
  cannot land in the persisted index. The instructor flag on the chunk is what
  drives authority weighting.

Copy `.env.example` to a local `.env` (also gitignored):

```bash
STAJ_ASISTAN_PRIVATE_CONFIG=data/private/config.json
TELEGRAM_BOT_TOKEN=
STAJ_ASISTAN_EMBEDDING_BACKEND=sentence-transformers
```

`STAJ_ASISTAN_INSTRUCTOR_ALIASES` still works for the sample-ingest path. The
private corpus builder reads aliases from the JSON config, not from that
variable.

## Build the masked index

Point the builder at local WhatsApp exports (`.txt` or `.zip`). Only text
entries inside a zip are read. The command prints aggregate counts only:

```bash
staj-asistan corpus build \
  --config data/private/config.json \
  --embedding-backend sentence-transformers \
  INPUT...
```

A successful build reports source, message, chunk and instructor-message counts
and `Privacy scan: clean`. It refuses to write the index if there is no
instructor chunk or if the masked chunks still contain personal-data patterns
or instructor aliases.

`data/index/` is gitignored. Delete that directory to drop the private index;
delete the SQLite file named in `queue_db` to drop the approval queue. Neither
belongs in git.

## Run the two processes

The poller and Streamlit share the ignored SQLite queue. They must be separate
processes: a Streamlit rerun must not reset the Telegram offset.

**Terminal 1 — poller** (holds the token for `getUpdates` only; it never calls
`sendMessage`):

```bash
set -a && source .env && set +a
staj-asistan telegram poll --config data/private/config.json
```

The worker long-polls, ignores chats that are not allowlisted, ignores bots and
non-text messages, masks each accepted body, triages it, and inserts a pending
row. The Telegram offset advances only after that handling succeeds.

**Terminal 2 — approval UI:**

```bash
set -a && source .env && set +a
streamlit run app.py
```

Open **Onay Kuyruğu**. Pending items show the masked incoming text, intent,
decision, evidence, citations and an editable draft.

- Tick the confirmation checkbox, then click **Onayla ve Telegram'a gönder**.
  The bot token is read from the environment only inside that click, not at
  page load.
- **Reddet** marks a pending item rejected. It is not sent.
- `DO_NOT_ANSWER` items cannot be sent. There is no send control for them.
- A definite Telegram rejection leaves the item `failed` so it can be approved
  again. An uncertain delivery leaves it `sending` and locked, because retrying
  could duplicate a message the group already received.

Without `STAJ_ASISTAN_PRIVATE_CONFIG` the app stays on the sample-data demo and
the queue view explains that Telegram is closed.

## Privacy boundary

- Raw exports stay under `data/private/` or outside the repo. They are
  gitignored.
- Masking runs before a queue row exists. The UI and the SQLite file hold
  masked text and pseudonyms, not display names.
- Chat IDs are stored as integers in the local queue because Telegram needs
  them to reply. They are operational identifiers, not names, and they must not
  be committed.
- Logs from the Telegram transport redact tokens and URLs.
- Do not screenshot the approval view while it is backed by the real corpus.

## Stop and uninstall locally

1. Stop the poller (`Ctrl+C`) and Streamlit.
2. Remove the bot from the Telegram group.
3. Delete `data/index/` (masked index + queue) and `data/private/config.json`.
4. Unset `TELEGRAM_BOT_TOKEN` and `STAJ_ASISTAN_PRIVATE_CONFIG`.
