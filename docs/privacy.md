# Privacy design

The corpus this assistant was built for contains real personal data: participant names, e-mail
addresses, phone numbers, private meeting links, and a roster of hundreds of people. The design
starts from the assumption that this data must never leave the machine and must never reach the
index.

## Principles

1. **Local by default.** No network call is required to ingest, index, retrieve or answer.
2. **Mask at ingestion, not at display.** Masking happens before a `Message` object exists, so
   every downstream artifact — chunks, embeddings, prompts, citations, the screen — is already
   clean. A display-time filter would leave the raw text sitting in the index, one bug away from
   exposure.
3. **Raw data cannot be committed.** Enforced by `.gitignore` patterns, not by remembering.
4. **Guarantees are tested.** Every masking rule has a test, and the committed sample files are
   scanned by the same guard that protects ingestion.

## What is removed

| Category | Detection | Replacement |
| --- | --- | --- |
| Personal e-mail | Address pattern | `[E-POSTA]` |
| Organisational e-mail | Address on a configured official domain | `[EĞİTMEN E-POSTA]` |
| Phone numbers | 10–15 digit candidate, separator or country code required, date shapes excluded | `[TELEFON]` |
| Private links | Teams, OneDrive, SharePoint, WhatsApp invites, roster hosts | `[ÖZEL LİNK]` |
| Roster tables | `Name <TAB> E-mail` line, consecutive rows collapsed | `[KATILIMCI LİSTESİ GİZLENDİ]` |
| Shared contact cards | `Name <address>` form | `[KİŞİ KARTI]` |
| `@handle` mentions | Handle pattern, program roles excluded | `[KULLANICI]` |
| Participant names | Sender field | `Katılımcı#a3f2` |

## Cohort-specific values are configuration, not code

A public repository should not name the resources it protects. The roster host and the
instructor's contact aliases are therefore not hardcoded; set them locally:

```bash
export STAJ_ASISTAN_PRIVATE_HOSTS="liste.example.net,intranet.example.org"
export STAJ_ASISTAN_INSTRUCTOR_ALIASES="ad soyad,ad soyad kurum"
```

Hosts listed in `STAJ_ASISTAN_PRIVATE_HOSTS` are added to the always-masked set. Mailbox names
and phone numbers are never shipped as defaults, because they belong to a person.

## What is deliberately kept

- **Public documentation links** (`learn.microsoft.com`, `github.com`, `pypi.org`). They are not
  personal data, and stripping them would make technical answers useless.
- **A generic instructor label in sample data.** Committed announcements use the role name
  ``Eğitmen`` / ``Instructor``, not a real person's display name. Authority weighting still
  works because it keys off that role, not off a private identity. On a real export, instructor
  recognition is supplied locally through ``STAJ_ASISTAN_INSTRUCTOR_ALIASES`` and is never
  shipped in this repository.
- **RFC 2606 documentation domains** (`example.com` and friends). They cannot belong to a real
  person, and keeping them lets the sample files show what a complete correction request looks
  like.
- **Message timestamps.** They carry no personal data and they drive recency weighting.

## Pseudonymisation

Participant names are replaced with a salted-hash pseudonym, so the same person keeps the same
label across all three groups and a conversation stays coherent. The mapping is one-way and is
never stored — the pseudonym is derived on demand and nothing writes a name-to-pseudonym table to
disk.

## Repository hygiene

Ignored: `data/private/`, `data/raw/`, `*.zip`, `*_chat.txt`, image/audio/video extensions, local
index files, `__pycache__`, `.DS_Store`, virtualenvs and `.env`.

Committed: `data/samples/` only, containing anonymised content written for demonstration.

Images are blocked everywhere except `screenshots/`, and that directory may only hold captures of
the application running on sample data. A screenshot of the assistant answering over the real
corpus would show participant messages on screen — the exact data the pipeline exists to strip —
and so would a chat, desktop or editor capture.

## Verification

Run on the three real exports (3 files, 2,108 messages, 990 chunks):

| Masked | Count |
| --- | --- |
| Roster rows | 1,464 |
| Private links | 601 |
| Participant names pseudonymised | 523 |
| `@handle` mentions | 389 |
| Personal e-mail addresses | 104 |
| Phone numbers | 78 |
| Organisational addresses | 53 |
| Shared contact cards | 3 |
| **Total** | **3,215** |

A scan of all 990 resulting chunks with independent patterns found **zero** e-mail addresses, zero
phone numbers and zero private links.

The counter above reports the text that was actually masked. An earlier version counted over the
raw file and inflated the number by more than 2,000, because the loose phone matcher was reading
chat timestamps as phone numbers — which also meant a message containing a deadline like
`30.09.2026 23:59` would have had the date replaced with `[TELEFON]`. Both are fixed, and both have
regression tests.

## Residual risks

Stated because pretending they do not exist is the actual risk:

- **Names in free-form prose survive.** Masking covers structured identifiers reliably. "Ali'ye
  sordum" is not detected, because there is no Turkish NER pass. Mitigated by the fact that
  first-name mentions in a group chat are weak identifiers, but it is a real gap.
- **Self-identification.** If someone writes their own full name into a message body, it stays.
- **Unknown private hosts.** New link shorteners are masked only after being added to the host
  list; unknown hosts on a common TLD are treated as public.
- **Pseudonyms are stable, therefore linkable.** That is intentional for conversation coherence,
  but it means an attacker with the salt and a candidate name list could confirm a guess. The salt
  is configurable for anyone who needs that closed.
- **The index is plaintext on disk.** It contains masked text only, but it is not encrypted; rely
  on full-disk encryption.
