# Answer quality review

Measured on HEAD `b38094c` against **sample data only** (`data/samples/`).
No application code was changed for this review. Push was not performed.

Runtime used for the measurements:

| Layer | Active backend |
| --- | --- |
| Embedding | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, dim=384) |
| Generation | **Extractive fallback** — `Alıntı modu (yerel model yok)` |
| Vector store | NumPy exact cosine |
| Index | 3 files, 46 messages, 37 instructor messages, 34 chunks |

`STAJ_ASISTAN_OFFLINE` was unset. Foundry Local was not reachable, so `resolve_llm_client()` fell through to `ExtractiveClient`. Embeddings still used sentence-transformers because that package is installed.

---

## 1. Why answers sometimes look like a quotation list

Generation is not a chat model in this session. `ExtractiveClient`:

1. splits each retrieved chunk into sentences,
2. scores a sentence by how many query tokens of length ≥ 4 appear as substrings,
3. takes the best sentence from the first few chunks, up to four bullets,
4. prefixes the whole answer with *“Yerel dil modeli çalışmadığı için kaynaklardan doğrudan alıntı yapıyorum (bu modda hiçbir cümle üretilmez)”*.

That is structural, not a prompt failure. Consequences:

- The answer is a **bullet list of snippets**, not a synthesised Turkish explanation.
- A markdown heading can win (`# Final Teslim Notları…`) because it contains `Final` / `Teslim` and is longer than 25 characters.
- The actual rule (“video + GitHub link, e-posta ile gönder”) can lose to a weaker sentence that shares a single keyword such as `gerekiyor`.
- Citation numbers follow retrieval order, not answer usefulness. Source `[2]` may never appear in the bullets even when it is the better announcement.

With Foundry Local running, `FoundryLocalClient` would write a short grounded paragraph. The checklist and correction modes would still compute their structure with rules; only the “kaynaklara göre not” / summary sentence would become fluent.

---

## 2. Retrieval pipeline

```text
query
  → Turkish stem + stopword filter (F5 truncation)
  → discriminative overlap gate (query must share a term with DF ≤ max_df_ratio)
  → dense search (top candidate_pool) + BM25 (top candidate_pool)
  → hybrid relevance = 0.65·dense + 0.35·lexical
  → multiplicative authority × recency
  → drop scores < min_score
  → drop near-duplicates (Jaccard ≥ 0.85), then MMR
  → top_k = 6
  → if empty: refuse, do not call the generator
```

Authority:

```text
multiplier = (1 + instructor_boost · instructor_ratio)
           · (1 − participant_penalty · (1 − instructor_ratio))
documents (role=DOCUMENT): multiplier = 1.0  (neither boost nor penalty)
```

Recency: half-life 45 days, weight 0.10.

MMR: `λ = 0.72`.

### Thresholds in this session

Embedding backend `sentence-transformers` → `min_score = 0.30`, `max_df_ratio = 0.5`.

Mode overrides (authority only; thresholds stay calibrated):

| Mode | instructor_boost | participant_penalty |
| --- | --- | --- |
| Instructor Q&A | 0.35 | 0.55 |
| Submission checklist | 0.35 | 0.60 |
| Correction analyzer | 0.35 | 0.60 |
| Technical help | 0.15 | 0.20 |

Evidence level (after retrieval): HIGH if top score ≥ `min_score × 2.2` and an instructor chunk is present; MEDIUM if top ≥ `min_score × 1.4`; else LOW. With `min_score=0.30` that is 0.66 / 0.42.

### When an unrelated snippet can still appear

1. **Lexical bleed.** A query token that is common in the corpus (`liste`, `GitHub`, `Foundry`, `Local`) pulls in chunks that share the word but not the intent. The DF gate only requires *one* distinctive term, not topical purity of every hit.
2. **Conversation windows.** An instructor reply is stored with the participant question that preceded it. A chunk about Foundry Local connection errors is tagged instructor and scores well whenever the query contains `Foundry`/`Local` *or* a shared stem from the surrounding window.
3. **top_k = 6 is wide** for a 34-chunk sample index. After the first 2–3 relevant hits, remaining slots fill with the next-best hybrid scores, which on this corpus are often neighbouring program topics.
4. **MMR prefers diversity**, so the 5th/6th results are *intentionally* less similar to the top hit — that is useful against reposts, harmful when the leftover topics are a different FAQ.
5. **Hashing backend** (not used in this measurement, but the default when sentence-transformers is missing) has `min_score=0.10` and is known to accept off-topic questions. The UI will look worse if that fallback is active.

Confirmed instance: **“Liste güncel değilse çalışmaya devam etmeli miyim?”** retrieved the Foundry Local “modele bağlanamıyor” window as source `[5]` (score 0.418, just above 0.30). The query does not mention Foundry. Likely path: lexical overlap on stems such as `çalış` / `devam` / generic program vocabulary inside a mixed conversation window, plus instructor boost keeping the chunk in the pool. Source `[6]` (README düzeni) is the same class of miss.

---

## 3. Demo question quality table

All five were asked with the **intended demo mode** (not `suggest_mode`). Generator was extractive except correction analyzer, which stays rule-based when the LLM is not generative.

| Question | Mode | Evidence | Sources | Quality |
| --- | --- | --- | --- | --- |
| Final tesliminde ne gerekiyor? | Instructor Q&A | medium (0.50) | 6 | **Weak.** Extractive quotes a heading and a certificate-email aside; the two-item rule (code link + 2 dk video, e-mail) is in the index but not in the answer body. |
| Liste güncel değilse çalışmaya devam etmeli miyim? | Instructor Q&A | medium (0.60) | 6 | **Mixed.** The correct instructor sentence *is* quoted (`Liste gecikmeli güncelleniyor, siz çalışmaya devam edin.`), but it is bullet 3, after two weaker “listeyi güncelledim / boşluk yakaladıkça” snippets. Sources 5–6 are off-topic (Foundry Local bağlantı, README). |
| GitHub repo hazır ama video çekmedim… | Submission checklist | high (0.77) | 6 | **Strong on the checklist, noisy on citations.** Ready/missing/next steps are correct and deterministic. The extractive “kaynak notu” then quotes roster GitHub-link lag and an Azure-student aside. |
| Listede projem yanlış… Bu mesaj yeterli mi? | Correction analyzer | medium (0.60) | 6 | **Strong on the workflow, noisy on citations.** JSON, missing `full_name`/`email`, `auto_apply: false`, and the draft e-mail are right. Retrieved sources are mostly Foundry Local technical notes because the query contains that project title. |
| Foundry Local çalışmazsa ne kontrol etmeliyim? | Technical help | medium (0.56) | 4 | **Incomplete.** The right chunk is source `[1]`, but extractive quoted the *participant question*, not the instructor’s port/endpoint checklist. `suggest_mode` would have sent this to Instructor Q&A (`çalışmazsa` is not in the technical hint list; `çalışmıyor` is). |

---

## 4. Per-question sources

### 4.1 Final tesliminde ne gerekiyor?

- **Mode:** Instructor Q&A (`suggest_mode` agrees).
- **Top sources:** (1) submission notes heading/overview, (2) 13.08 “videoları seyredip sertifika”, (3) 22.07 “bana sadece linki gerekiyor”, (4) 22.06 “olur tabii… bitirmeniz”, (5) submission notes certificate section, (6) 18.08 certificate batches.
- **Irrelevant?** `[4]` is a generic “keep going” reply, not a submission spec. `[2]` and `[6]` are certificate process, adjacent but not “what to submit”. The 16.07 / 01.08 announcements that actually list *video + source link* did not make top 6.
- **User-facing answer?** No. A participant would not walk away knowing the two required artefacts.

### 4.2 Liste güncel değilse çalışmaya devam etmeli miyim?

- **Mode:** Instructor Q&A (`suggest_mode` agrees).
- **Top sources:** (1) 10.08 “Listeyi güncelledim…”, (2) 17.06 pinli liste / düzeltme ricasi, (3) **17.06 Q+A: devam edin** (correct), (4) 11.08 “bu liste sadece benim takibim için”, (5) **06.07 Foundry Local bağlanamıyor**, (6) 09.07 README düzeni.
- **Irrelevant?** **Yes — `[5]` and `[6]`.** `[5]` is exactly the Technical Help / connection-error contamination called out in the review request. `[1]` and `[2]` are on-roster but do not answer “should I keep working?”.
- **User-facing answer?** Partially. The needed sentence is present, buried as the third bullet, surrounded by list-admin chatter and two off-topic chunks.

### 4.3 GitHub repo hazır ama video çekmedim…

- **Mode:** Submission checklist (`suggest_mode` agrees).
- **Top sources:** (1) 10.08 roster/GitHub-link lag, (2) 16.07 GitHub + short video (on-point), (3) submission notes “video çekmedim → eksik”, (4) GitHub student / Azure aside, (5) GitHub vs public link, (6) 13.08 bitirme maili.
- **Irrelevant?** `[1]` and `[4]` are GitHub-word matches, not “is my submission complete?”. Checklist body does not depend on them.
- **User-facing answer?** Yes for the structured part (video missing, e-mail unspecified). The quotation appendix undercuts that clarity.

### 4.4 Listede projem yanlış… Foundry Local olarak güncellenmesini…

- **Mode:** Correction analyzer (`suggest_mode` agrees; `listede` / `güncellenmesini` fire first).
- **Top sources:** (1) 13.07 why Foundry Local, (2) 17.06 list correction process, (3) 10.08 list update, (4) Foundry Local CUDA/CPU SDK note, (5) local RAG architecture, (6) Foundry Local bağlanamıyor.
- **Irrelevant?** **`[1]`, `[4]`, `[5]`, `[6]`** are project-name / Foundry lexical hits. `[2]` is the only source that actually describes how list corrections work. This mode does not need those sources for the JSON; they only appear in the UI citation panel.
- **User-facing answer?** Yes. The recommendation is explicit: WhatsApp is not enough; send name + e-mail. Citations are misleading if the user reads them as “evidence for the JSON”.

### 4.5 Foundry Local çalışmazsa ne kontrol etmeliyim?

- **Mode used in this measurement:** Technical Help (demo intent). **UI hint would be Instructor Q&A** because `_MODE_HINTS` has `çalışmıyor` but not `çalışmazsa` / `bağlanamıyor` / `foundry`.
- **Top sources:** (1) 06.07 bağlanamıyor Q+A (correct window), (2) 13.07 why Foundry Local, (3) local RAG architecture, (4) 03.08 CUDA/CPU SDK note.
- **Irrelevant?** `[2]` and `[3]` are neighbouring Foundry topics, not a troubleshooting checklist. `[4]` is useful only if the failure mode is silent CPU fallback.
- **User-facing answer?** No. Extractive emitted the participant’s question as the first bullet and never quoted “servis ayakta / port / OpenAI-uyumlu endpoint / modeli çalıştırın”.

---

## 5. Five concrete improvements

1. **Run Foundry Local for demo answers** (`foundry service start` + `foundry model run phi-4-mini`). Extractive mode is the main reason answers look like a dump of quotes. This is configuration, not a code change, and is the single largest quality jump for Q&A.
2. **Query-side topic filter for Instructor Q&A.** If the query is about roster/devam (`liste`, `güncel`, `devam`), drop chunks whose distinctive terms are only technical (`endpoint`, `cuda`, `bağlanamıyor`, `embedding`) unless those terms are also in the query. Directly targets the 4.2 leak.
3. **Tighter `top_k` for Q&A (3 instead of 6)** once sentence-transformers is active, or a second-pass “intent overlap” so the 4th–6th slots cannot be a different FAQ. MMR should not be allowed to fill with a different program topic.
4. **Extractive sentence picker:** skip headings and skip a sentence that is a near-copy of the user question; prefer instructor sentences; require at least one query stem that is not a corpus-wide word. Unblocks Q1 and Q5 even when Foundry Local is down.
5. **Mode hints:** add `çalışmazsa`, `bağlanamıyor`, `foundry local`, `endpoint` already exists — also `model`. Keep correction analyzer from retrieving Foundry architecture chunks by down-weighting the project-title tokens once `intent=list_correction` is known (the structured path already has the project; retrieval is only for the citation panel).

---

## 6. Improvement plan

### Quick fixes (about 30 minutes)

- Document in the UI sidebar, in one line, that answers are extractive until Foundry Local is up — users currently read the quotation disclaimer as a bug.
- Extend `_MODE_HINTS` technical keywords: `çalışmazsa`, `bağlanamıyor`, `foundry`.
- Extractive: ignore lines starting with `#` and sentences that are >80% the same tokens as the question.
- Instructor Q&A: `top_k=4` for sentence-transformers (leave hashing at 6 if needed for recall).

### Medium fixes (1–2 hours)

- Intent-aware retrieval filter (section 5.2) with a test: “Liste güncel değilse…” must not return the 06.07 Foundry connection chunk.
- Prefer instructor-only sentences in extractive output; if an instructor sentence exists in the chunk, never quote the participant question.
- For correction analyzer, retrieve with a rewritten query that strips the project catalog title (`Foundry Local`) and keeps `liste` / `düzeltme` / `e-posta`, *or* hide citations when `auto_apply` is false and the JSON already carries the recommendation.
- Add a regression test that the extractive answer to “Final tesliminde ne gerekiyor?” contains `video` and (`GitHub` or `link`).

### Optional polish (if time remains)

- Start Foundry Local in the demo script and show the same five questions with generative answers vs extractive, so the presentation can name the fallback honestly.
- Cross-encoder rerank of the candidate pool.
- Query rewriting: expand “tesliminde ne gerekiyor” → `video, GitHub, e-posta, README` before retrieval so the 16.07 / 01.08 announcements enter top_k.
- Per-mode evidence copy: medium + extractive should not read as confidently as a Foundry Local paragraph.

---

## 7. Screenshots

Captured from Streamlit on sample data, viewport of the app only. Nested under `screenshots/review/`, which is git-ignored (`*.png` except `screenshots/*.png`). **Not committed.**

| File | What it shows |
| --- | --- |
| `screenshots/review/01-sample-indexed.png` | Sample index loaded (3 files, no private paths) |
| `screenshots/review/02-liste-devam.png` | “Liste güncel değilse…” answer + citation panel (off-topic Foundry chunk visible) |
| `screenshots/review/03-checklist.png` | Checklist mode on the GitHub/video question |
| `screenshots/review/04-correction.png` | Correction JSON + missing identity |
| `screenshots/review/05-foundry-help.png` | **Not captured.** Streamlit’s sidebar overlay intercepted the Technical Help radio; quality for this question is fully documented from the CLI run in §4.5. |

---

## 8. Recommendation before polish

Do not treat this as “the model is bad”. Retrieval already has the right instructor sentence for the roster question; generation is quoting the wrong slice of a too-wide hit list, and Foundry Local is not running. Fix order: **Foundry Local on for the demo**, then **stop off-topic chunks on roster queries**, then **extractive hardening** so the offline path is still presentable.
