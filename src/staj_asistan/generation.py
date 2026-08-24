"""Answer generation.

Primary runtime is **Microsoft Foundry Local**, reached through its
OpenAI-compatible ``/v1/chat/completions`` endpoint. Discovery order:

1. the official ``foundry-local-sdk`` if it is installed (it knows the port and
   the resolved model id for the current machine)
2. ``FOUNDRY_LOCAL_ENDPOINT`` / ``FOUNDRY_LOCAL_MODEL`` environment variables
3. the default local endpoint

If no local model is running, the assistant does **not** fall back to a cloud
API and does not stay silent: it switches to an extractive generator that quotes
the retrieved sources directly. That keeps the demo reproducible offline and,
more importantly, makes hallucination structurally impossible in that mode.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from .retriever import content_tokens

DEFAULT_ENDPOINT = "http://localhost:5273/v1"
DEFAULT_MODEL = "phi-4-mini"


class LLMUnavailable(RuntimeError):
    """Raised when a chat backend cannot be reached, so the chain can fall through."""


class LLMClient(ABC):
    """Minimal chat interface used by the pipeline."""

    name: str = "unknown"
    #: True when the client can produce free-form text rather than quoted extracts.
    generative: bool = True

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str: ...

    @property
    def description(self) -> str:
        return self.name


class FoundryLocalClient(LLMClient):
    """Chat completions from a locally running Foundry Local service."""

    name = "foundry-local"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.2,
    ) -> None:
        import requests

        self._requests = requests
        self.timeout = timeout
        self.temperature = temperature
        sdk_endpoint, sdk_model = _discover_via_sdk(model)
        self.endpoint = (
            endpoint
            or sdk_endpoint
            or os.getenv("FOUNDRY_LOCAL_ENDPOINT", DEFAULT_ENDPOINT)
        ).rstrip("/")
        self.model = model or sdk_model or os.getenv("FOUNDRY_LOCAL_MODEL") or self._first_model()

    def _first_model(self) -> str:
        """Ask the running service which model is loaded, instead of guessing."""
        try:
            response = self._requests.get(f"{self.endpoint}/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            if models:
                return models[0]["id"]
        except Exception as exc:  # noqa: BLE001 - any failure means "not running"
            raise LLMUnavailable(
                f"Foundry Local is not reachable at {self.endpoint}. "
                "Start it with 'foundry service start' and load a model with "
                "'foundry model run phi-4-mini'."
            ) from exc
        raise LLMUnavailable(
            f"Foundry Local is running at {self.endpoint} but no model is loaded. "
            "Run 'foundry model run phi-4-mini'."
        )

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
        try:
            response = self._requests.post(
                f"{self.endpoint}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"Foundry Local request failed: {exc}") from exc

    @property
    def description(self) -> str:
        return f"Foundry Local · {self.model}"


class OpenAICompatibleClient(FoundryLocalClient):
    """Any other local OpenAI-compatible server (Ollama, vLLM, LM Studio)."""

    name = "openai-compatible"

    def __init__(self, endpoint: str, model: str | None = None, **kwargs) -> None:
        super().__init__(endpoint=endpoint, model=model, **kwargs)

    @property
    def description(self) -> str:
        return f"OpenAI uyumlu yerel uç nokta · {self.model}"


class ExtractiveClient(LLMClient):
    """Deterministic, quote-only generator used when no local model is available.

    It cannot invent anything: the output is assembled exclusively from sentences
    that already exist in the retrieved sources. The layout is a short answer,
    an action list, quoted evidence and source markers — not a raw quote dump.
    """

    name = "extractive"
    generative = False

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n")
    _HEADER_RE = re.compile(
        r"^\[\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\]\s*[^:\n]+:\s*"
    )
    _ACTION_RE = re.compile(
        r"\b(edin|atın|gönderin|kontrol edin|yükleyin|koyun|çalıştırın|"
        r"emin olun|bekleyin|ekleyin|yazın|seçin|devam edin)\b",
        re.IGNORECASE,
    )
    _WEAK_QUERY_STEMS = frozenset({"gerek", "olmal", "lazım", "nedir", "nasil", "nasıl"})

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
        question, sources = _split_prompt(user_prompt)
        question_stems = {
            stem for stem in content_tokens(question) if stem not in self._WEAK_QUERY_STEMS
        }
        ranked: list[tuple[int, str, int]] = []
        for index, body in sources:
            for sentence, score in self._scored_sentences(body, question_stems):
                ranked.append((score, sentence, index))
        ranked.sort(key=lambda item: -item[0])

        if not ranked:
            return (
                "Kısa cevap:\n"
                "Kaynaklarda bu soruya doğrudan karşılık gelen bir cümle bulamadım.\n\n"
                "Ne yapmalısın:\n"
                "- Aşağıdaki kaynak parçalarını kendin inceleyebilirsin.\n\n"
                "Kaynaklara göre kanıt:\n"
                "- Kaynaklarda doğrudan alıntılanacak bir cümle yok.\n\n"
                "Kaynaklar:\n"
                + (", ".join(f"[{index}]" for index, _ in sources) if sources else "—")
            )

        seen: set[str] = set()
        unique: list[tuple[int, str, int]] = []
        for score, sentence, index in ranked:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append((score, sentence, index))

        short_sentence, short_index = unique[0][1], unique[0][2]
        actions: list[tuple[str, int]] = [(short_sentence, short_index)]
        for _, sentence, index in unique[1:]:
            if not self._ACTION_RE.search(sentence):
                continue
            if any(sentence == existing for existing, _ in actions):
                continue
            actions.append((sentence, index))
            if len(actions) >= 3:
                break

        evidence = unique[:3]
        used = sorted({short_index, *(index for _, index in actions), *(item[2] for item in evidence)})

        action_lines = "\n".join(f"- {sentence} [{index}]" for sentence, index in actions)
        evidence_lines = "\n".join(f"- {sentence} [{index}]" for _, sentence, index in evidence)
        markers = ", ".join(f"[{index}]" for index in used)
        return (
            f"Kısa cevap:\n{short_sentence} [{short_index}]\n\n"
            f"Ne yapmalısın:\n{action_lines}\n\n"
            f"Kaynaklara göre kanıt:\n{evidence_lines}\n\n"
            f"Kaynaklar:\n{markers}"
        )

    def _scored_sentences(
        self, text: str, question_stems: set[str]
    ) -> list[tuple[str, int]]:
        scored: list[tuple[str, int]] = []
        for raw in self._SENTENCE_RE.split(text):
            original = raw.strip()
            sentence = self._clean_sentence(original)
            if original.startswith("#") or not self._usable(sentence):
                continue
            sent_stems = set(content_tokens(sentence))
            score = len(question_stems & sent_stems)
            if "eğitmen:" in original.lower():
                score += 2
            if self._ACTION_RE.search(sentence):
                score += 2
            if sentence.endswith("?"):
                score -= 2
            if question_stems and sent_stems:
                overlap = len(question_stems & sent_stems) / len(question_stems | sent_stems)
                if overlap >= 0.65:
                    score -= 3
            if score <= 0:
                continue
            scored.append((sentence, score))
        return scored

    def _clean_sentence(self, sentence: str) -> str:
        sentence = sentence.strip().lstrip("-*• ").strip()
        sentence = re.sub(r"^#+\s+", "", sentence)
        sentence = self._HEADER_RE.sub("", sentence)
        return re.sub(r"\s+", " ", sentence).strip()

    @staticmethod
    def _usable(sentence: str) -> bool:
        if len(sentence) < 25:
            return False
        if sentence.startswith("#"):
            return False
        lowered = sentence.lower()
        if lowered.startswith("bu dosya") or lowered.startswith("örnek "):
            return False
        return True

    @property
    def description(self) -> str:
        return "Alıntı modu (yerel model yok)"


def _split_prompt(user_prompt: str) -> tuple[str, list[tuple[int, str]]]:
    """Recover the question and the numbered sources from an assembled prompt."""
    question = user_prompt.rsplit("\n", 2)[0].split("\n")[-1] if user_prompt else ""
    for marker in ("SORU\n", "TEKNİK SORU\n", "KATILIMCININ BİLDİRDİĞİ TESLİM DURUMU\n", "KATILIMCININ DÜZELTME MESAJI\n"):
        if marker in user_prompt:
            question = user_prompt.split(marker, 1)[1].split("\n\nCEVAP")[0].strip()
            break
    sources: list[tuple[int, str]] = []
    for match in re.finditer(r"^\[(\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\n[A-ZÇĞİÖŞÜ]{3,}\n|\Z)", user_prompt, re.S | re.M):
        sources.append((int(match.group(1)), match.group(2).strip()))
    return question, sources


def _discover_via_sdk(alias: str | None) -> tuple[str | None, str | None]:
    """Use the official Foundry Local SDK when present; stay silent when not."""
    try:
        from foundry_local import FoundryLocalManager
    except Exception:  # noqa: BLE001 - SDK is optional
        return None, None
    try:
        manager = FoundryLocalManager(alias or os.getenv("FOUNDRY_LOCAL_MODEL", DEFAULT_MODEL))
        model_info = manager.get_model_info(alias or os.getenv("FOUNDRY_LOCAL_MODEL", DEFAULT_MODEL))
        return manager.endpoint, getattr(model_info, "id", None)
    except Exception:  # noqa: BLE001 - service not running or model unavailable
        return None, None


def resolve_llm_client(preference: str | None = None) -> LLMClient:
    """Pick a chat backend.

    ``STAJ_ASISTAN_LLM`` accepts ``auto`` (default), ``foundry``, ``extractive``
    or an ``http://...`` endpoint of any OpenAI-compatible local server.
    ``STAJ_ASISTAN_OFFLINE=1`` forces the extractive client.
    """
    choice = (preference or os.getenv("STAJ_ASISTAN_LLM", "auto")).strip().lower()
    if os.getenv("STAJ_ASISTAN_OFFLINE", "").strip() in {"1", "true", "yes"}:
        return ExtractiveClient()
    if choice == "extractive":
        return ExtractiveClient()
    if choice.startswith("http"):
        try:
            return OpenAICompatibleClient(endpoint=choice)
        except LLMUnavailable:
            return ExtractiveClient()
    if choice in {"auto", "foundry", "foundry-local"}:
        try:
            return FoundryLocalClient()
        except LLMUnavailable:
            if choice == "auto":
                return ExtractiveClient()
            raise
    raise ValueError(f"Unknown LLM backend '{choice}' (use 'auto', 'foundry', 'extractive' or a URL)")
