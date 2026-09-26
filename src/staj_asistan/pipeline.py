"""The assistant: wires ingestion, retrieval, workflows and generation together.

Answer contract, enforced here rather than trusted to the prompt:

* if retrieval finds nothing above the calibrated threshold, no model is called
  at all and the assistant says the sources do not cover the question;
* citations are built from the chunks that were actually retrieved;
* citation markers pointing at sources that do not exist are stripped;
* the checklist and correction modes compute their structure deterministically,
  and the model may only rephrase it.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from .chunking import DEFAULT_CHUNKING, ChunkingConfig
from .embeddings import EmbeddingBackend, resolve_embedding_backend
from .generation import ExtractiveClient, LLMClient, LLMUnavailable, resolve_llm_client
from .ingestion import (
    IngestedSource,
    ingest_bytes,
    ingest_directory,
    ingest_path,
    ingest_text,
    sample_directory,
)
from .models import (
    Answer,
    AssistantMode,
    ChunkCategory,
    Citation,
    EvidenceLevel,
    IndexStats,
    ScoredChunk,
)
from .privacy import DEFAULT_POLICY, PrivacyPolicy
from .prompts import (
    NO_EVIDENCE_ANSWER,
    SYSTEM_PROMPTS,
    build_correction_summary_prompt,
    build_user_prompt,
)
from .retriever import RetrievalConfig, Retriever
from .topics import filter_scored_chunks, infer_query_topic, is_programme_topic
from .vector_store import VectorStore, build_vector_store
from .whatsapp_parser import DEFAULT_INSTRUCTOR, InstructorIdentity
from .workflows import (
    CHANNEL_RULE_LINES,
    SUBMISSION_RULE_LINES,
    analyze_correction_request,
    analyze_submission,
    format_channel_rules,
    format_submission_rules,
    is_submission_channel_query,
    is_submission_requirement_query,
    resolve_mode,
)

SNIPPET_LENGTH = 320

#: Per-mode retrieval tweaks. Technical help values participant experience reports
#: more highly than rule questions do, so its authority penalty is softer.
MODE_RETRIEVAL_OVERRIDES: dict[AssistantMode, dict] = {
    AssistantMode.INSTRUCTOR_QA: {"instructor_boost": 0.35, "participant_penalty": 0.55},
    AssistantMode.SUBMISSION_CHECKLIST: {"instructor_boost": 0.35, "participant_penalty": 0.6},
    AssistantMode.CORRECTION_ANALYZER: {"instructor_boost": 0.35, "participant_penalty": 0.6},
    AssistantMode.TECHNICAL_HELP: {"instructor_boost": 0.15, "participant_penalty": 0.2},
}


class Assistant:
    """Stateful assistant holding one index and one set of backends."""

    def __init__(
        self,
        policy: PrivacyPolicy = DEFAULT_POLICY,
        chunking: ChunkingConfig = DEFAULT_CHUNKING,
        embedding_backend: EmbeddingBackend | None = None,
        llm: LLMClient | None = None,
        store: VectorStore | None = None,
        retrieval: RetrievalConfig | None = None,
        instructor: InstructorIdentity = DEFAULT_INSTRUCTOR,
    ) -> None:
        self.policy = policy
        self.chunking = chunking
        self.instructor = instructor
        self.embedding_backend = embedding_backend or resolve_embedding_backend()
        self.llm = llm or resolve_llm_client()
        self.store = store or build_vector_store("numpy")
        self.retrieval = retrieval or RetrievalConfig.calibrated_for(self.embedding_backend.name)
        self.retriever = Retriever(self.store, self.embedding_backend, self.retrieval)
        self._sources: list[IngestedSource] = []

    # ---------------------------------------------------------------- ingestion

    def add_source(self, source: IngestedSource) -> int:
        """Chunk, embed and index one source. Returns the number of chunks added."""
        chunks = source.to_chunks(self.chunking)
        if chunks:
            vectors = self.embedding_backend.embed_documents([c.text for c in chunks])
            self.store.add(chunks, vectors)
            self.retriever.invalidate()
        self._sources.append(source)
        return len(chunks)

    def add_sources(self, sources: list[IngestedSource]) -> int:
        return sum(self.add_source(source) for source in sources)

    def ingest_text(self, text: str, source: str) -> int:
        return self.add_source(ingest_text(text, source, self.policy, self.instructor))

    def ingest_file(self, path) -> int:
        return self.add_sources(ingest_path(path, self.policy, self.instructor))

    def ingest_upload(self, data: bytes, filename: str) -> int:
        return self.add_sources(ingest_bytes(data, filename, self.policy, self.instructor))

    def load_samples(self, directory=None) -> int:
        """Index the anonymised sample data that ships with the repository."""
        return self.add_sources(
            ingest_directory(directory or sample_directory(), self.policy, self.instructor)
        )

    def reset(self) -> None:
        self.store.clear()
        self.retriever.invalidate()
        self._sources = []

    @property
    def stats(self) -> IndexStats:
        return IndexStats(
            file_count=len(self._sources),
            message_count=sum(s.message_count for s in self._sources),
            instructor_message_count=sum(s.instructor_message_count for s in self._sources),
            chunk_count=len(self.store),
            masked_entity_count=sum(s.masked_entities for s in self._sources),
            sources=tuple(s.source for s in self._sources),
            embedding_backend=self.embedding_backend.description,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.store) == 0

    # ---------------------------------------------------------------- answering

    def _retrieve(self, query: str, mode: AssistantMode, top_k: int | None) -> list[ScoredChunk]:
        # Start from the assistant's calibrated config so backend thresholds survive,
        # then apply only the mode-specific authority weights on top.
        config = replace(self.retrieval, **MODE_RETRIEVAL_OVERRIDES.get(mode, {}))
        retriever = Retriever(self.store, self.embedding_backend, config)
        return filter_scored_chunks(retriever.retrieve(query, top_k=top_k), query, mode)

    def _prune_results(
        self,
        results: list[ScoredChunk],
        evidence: EvidenceLevel,
        query: str,
        mode: AssistantMode,
    ) -> list[ScoredChunk]:
        """Keep the citation panel small: top 2 on high evidence, otherwise at most 3.

        Same-topic chunks stay ahead of merely allowed neighbours so a teslim
        question is not answered from a leftover roster snippet.
        """
        if not results:
            return []
        topic = infer_query_topic(query, mode)
        if topic is not ChunkCategory.GENERAL:
            primary = [item for item in results if item.chunk.category is topic]
            secondary = [item for item in results if item.chunk.category is not topic]
            results = primary + secondary or results
        floor = self.retrieval.min_score * 1.4
        strong = [item for item in results if item.score >= floor]
        pool = strong or results[:1]
        cap = 2 if evidence is EvidenceLevel.HIGH else 3
        return pool[:cap]

    def _evidence_level(self, results: list[ScoredChunk]) -> EvidenceLevel:
        if not results:
            return EvidenceLevel.NONE
        top = results[0].score
        threshold = self.retrieval.min_score
        has_instructor = any(r.chunk.is_instructor for r in results)
        if top >= threshold * 2.2 and has_instructor:
            return EvidenceLevel.HIGH
        if top >= threshold * 1.4:
            return EvidenceLevel.MEDIUM
        return EvidenceLevel.LOW

    @staticmethod
    def _citations(results: list[ScoredChunk]) -> list[Citation]:
        citations = []
        for index, scored in enumerate(results, start=1):
            chunk = scored.chunk
            snippet = chunk.text.strip()
            if len(snippet) > SNIPPET_LENGTH:
                snippet = snippet[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + " …"
            citations.append(
                Citation(
                    index=index,
                    label=chunk.label,
                    source=chunk.source,
                    role=chunk.role,
                    snippet=snippet,
                    score=scored.score,
                )
            )
        return citations

    @staticmethod
    def _strip_invalid_markers(text: str, source_count: int) -> str:
        """Remove citation markers that point beyond the sources we actually sent."""
        return re.sub(
            r"\[(\d+)\]",
            lambda m: m.group(0) if 1 <= int(m.group(1)) <= source_count else "",
            text,
        )

    def _generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> tuple[str, str]:
        """Run the LLM, degrading to the extractive client if it disappears mid-session."""
        try:
            return self.llm.complete(system_prompt, user_prompt, max_tokens), self.llm.description
        except LLMUnavailable:
            fallback = ExtractiveClient()
            self.llm = fallback
            return fallback.complete(system_prompt, user_prompt, max_tokens), fallback.description

    def ask(
        self, question: str, mode: AssistantMode = AssistantMode.INSTRUCTOR_QA, top_k: int | None = None
    ) -> Answer:
        """Answer a question in the requested mode."""
        question = question.strip()
        if not question:
            return Answer(
                text="Lütfen bir soru veya durum yaz.",
                mode=mode,
                evidence=EvidenceLevel.NONE,
                generator="-",
            )

        mode = resolve_mode(question, mode)

        results = self._retrieve(question, mode, top_k)
        evidence = self._evidence_level(results)
        # An off-topic question can still overlap the corpus lexically, and the
        # deterministic modes would answer it from their rule cards regardless of
        # retrieval. Refuse before dispatching so every mode refuses identically.
        if evidence is not EvidenceLevel.HIGH and not is_programme_topic(question):
            return self._refuse(mode)
        results = self._prune_results(results, evidence, question, mode)
        if mode is AssistantMode.SUBMISSION_CHECKLIST:
            return self._answer_checklist(question, results, mode, evidence)
        if mode is AssistantMode.CORRECTION_ANALYZER:
            return self._answer_correction(question, results, mode, evidence)
        return self._answer_qa(question, results, mode, evidence)

    def _refuse(self, mode: AssistantMode) -> Answer:
        """A refusal, and nothing else: no snippet, no citation, no structured card."""
        return Answer(
            text=NO_EVIDENCE_ANSWER,
            mode=mode,
            evidence=EvidenceLevel.NONE,
            generator=self.llm.description,
        )

    def _answer_qa(self, question, results, mode, evidence: EvidenceLevel) -> Answer:
        if not results:
            return self._refuse(mode)
        raw, generator = self._generate(
            SYSTEM_PROMPTS[mode], build_user_prompt(question, results, mode)
        )
        text = self._strip_invalid_markers(raw, len(results))
        if evidence is EvidenceLevel.LOW and "kaynaklarda net değil" not in text.lower():
            text = "Bu konu kaynaklarda net değil.\n\n" + text
        return Answer(
            text=text,
            mode=mode,
            evidence=evidence,
            citations=self._citations(results),
            retrieved=results,
            generator=generator,
            grounded=bool(re.search(r"\[\d+\]", text)) or not self.llm.generative,
        )

    def _answer_checklist(self, status_text, results, mode, evidence: EvidenceLevel) -> Answer:
        if is_submission_channel_query(status_text):
            return self._answer_channel_rules(status_text, results, mode, evidence)
        if is_submission_requirement_query(status_text):
            return self._answer_submission_rules(status_text, results, mode, evidence)

        checklist = analyze_submission(status_text)
        body = checklist.as_text()
        generator = "kural tabanlı"

        if results:
            raw, generator = self._generate(
                SYSTEM_PROMPTS[mode],
                build_user_prompt(status_text, results, mode),
                max_tokens=450,
            )
            note = self._strip_invalid_markers(raw, len(results))
            body = f"{body}\n\n**Kaynaklara göre not**\n{note}"
        else:
            body = (
                f"{body}\n\n_Not: İndekste teslim kurallarını içeren kaynak bulunamadı; "
                "yukarıdaki kontrol listesi program kurallarından türetilmiş varsayılan kurallara dayanıyor._"
            )

        if evidence is EvidenceLevel.LOW and "kaynaklarda net değil" not in body.lower():
            body = f"{body}\n\n_Bu konu kaynaklarda net değil; kontrol listesi kurallardan, notlar ise en yakın eşleşmelerden üretildi._"

        return Answer(
            text=body,
            mode=mode,
            evidence=evidence if results else EvidenceLevel.LOW,
            citations=self._citations(results),
            retrieved=results,
            structured=checklist.as_dict(),
            generator=generator,
        )

    def _answer_submission_rules(self, question, results, mode, evidence: EvidenceLevel) -> Answer:
        body = format_submission_rules()
        generator = "kural tabanlı"
        structured = {"tur": "teslim_kurallari", "gerekler": list(SUBMISSION_RULE_LINES)}

        if results:
            raw, generator = self._generate(
                SYSTEM_PROMPTS[mode],
                build_user_prompt(question, results, mode),
                max_tokens=450,
            )
            note = self._strip_invalid_markers(raw, len(results))
            body = f"{body}\n\n**Kaynaklara göre not**\n{note}"
        else:
            body = (
                f"{body}\n\n_Not: İndekste teslim kurallarını içeren kaynak bulunamadı; "
                "yukarıdaki maddeler program kurallarından türetilmiştir._"
            )

        if evidence is EvidenceLevel.LOW and "kaynaklarda net değil" not in body.lower():
            body = f"{body}\n\n_Bu konu kaynaklarda net değil; kurallar program özetinden, notlar ise en yakın eşleşmelerden üretildi._"

        return Answer(
            text=body,
            mode=mode,
            evidence=evidence if results else EvidenceLevel.LOW,
            citations=self._citations(results),
            retrieved=results,
            structured=structured,
            generator=generator,
            grounded=bool(re.search(r"\[\d+\]", body)) or not self.llm.generative,
        )

    def _answer_channel_rules(self, question, results, mode, evidence: EvidenceLevel) -> Answer:
        body = format_channel_rules()
        generator = "kural tabanlı"
        structured = {"tur": "teslim_kanali", "gerekler": list(CHANNEL_RULE_LINES)}

        if results:
            raw, generator = self._generate(
                SYSTEM_PROMPTS[mode],
                build_user_prompt(question, results, mode),
                max_tokens=450,
            )
            note = self._strip_invalid_markers(raw, len(results))
            body = f"{body}\n\n**Kaynaklara göre not**\n{note}"
        else:
            body = (
                f"{body}\n\n_Not: İndekste kanal kuralını içeren kaynak bulunamadı; "
                "yukarıdaki maddeler program kurallarından türetilmiştir._"
            )

        if evidence is EvidenceLevel.LOW and "kaynaklarda net değil" not in body.lower():
            body = f"{body}\n\n_Bu konu kaynaklarda net değil; kurallar program özetinden, notlar ise en yakın eşleşmelerden üretildi._"

        return Answer(
            text=body,
            mode=mode,
            evidence=evidence if results else EvidenceLevel.LOW,
            citations=self._citations(results),
            retrieved=results,
            structured=structured,
            generator=generator,
            grounded=bool(re.search(r"\[\d+\]", body)) or not self.llm.generative,
        )

    @staticmethod
    def _format_missing(analysis) -> str:
        labels = {
            "full_name": "ad-soyad",
            "email": "e-posta",
            "requested_change": "istenen değişiklik",
        }
        missing = analysis.missing_required_info
        if not missing:
            return "yok, istek eksiksiz görünüyor"
        return ", ".join(labels.get(field, field) for field in missing)

    def _answer_correction(self, message, results, mode, evidence: EvidenceLevel) -> Answer:
        analysis = analyze_correction_request(message)
        structured = analysis.as_dict()
        generator = "kural tabanlı"
        summary = analysis.recommendation

        if results and self.llm.generative:
            raw, generator = self._generate(
                SYSTEM_PROMPTS[mode],
                build_correction_summary_prompt(structured, results),
                max_tokens=400,
            )
            summary = self._strip_invalid_markers(raw, len(results))

        if evidence is EvidenceLevel.LOW and "kaynaklarda net değil" not in summary.lower():
            summary = "Kaynaklarda bu düzeltme kuralı net değil. " + summary

        # Stated up front rather than left implicit in the JSON. The question behind
        # every one of these messages is "did this fix it?", and the answer is no.
        header = (
            "**Bu bir düzeltme isteği**\n"
            "- Asistan listeyi değiştiremez; düzeltme otomatik olarak uygulanmaz.\n"
            "- Değişikliği yalnızca eğitmen yapabilir, yani insan onayı gerekir.\n"
            "- İsteği e-posta ile gönder; WhatsApp mesajı tek başına yeterli değil.\n"
            f"- Eksik bilgi: {self._format_missing(analysis)}"
        )

        body = (
            f"{header}\n\n{summary}\n\n"
            "**Yapılandırılmış çıktı**\n```json\n"
            f"{json.dumps(structured, ensure_ascii=False, indent=2)}\n```\n\n"
            "**E-posta taslağı (gönderilmedi, sadece öneri)**\n```\n"
            f"{analysis.suggested_email}\n```"
        )
        return Answer(
            text=body,
            mode=mode,
            evidence=evidence if results else EvidenceLevel.LOW,
            citations=self._citations(results),
            retrieved=results,
            structured=structured,
            generator=generator,
        )


def build_default_assistant(**kwargs) -> Assistant:
    """Convenience constructor used by the UI and the CLI."""
    return Assistant(**kwargs)
