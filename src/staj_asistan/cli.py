"""Command-line entry point.

Useful for demos and for checking the pipeline without starting Streamlit::

    staj-asistan ask "Final tesliminde ne gerekiyor?"
    staj-asistan ask "GitHub repo hazır ama video çekmedim" --mode submission_checklist
    staj-asistan stats --data data/private
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import build_private_index
from .embeddings import resolve_embedding_backend
from .ingestion import sample_directory
from .models import AssistantMode
from .pipeline import Assistant
from .private_config import PrivateConfig


def _build_assistant(data_dir: str | None) -> Assistant:
    assistant = Assistant()
    directory = Path(data_dir) if data_dir else sample_directory()
    assistant.load_samples(directory)
    return assistant


def _print_answer(answer, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "mode": answer.mode.value,
                    "evidence": answer.evidence.value,
                    "generator": answer.generator,
                    "text": answer.text,
                    "structured": answer.structured,
                    "citations": [
                        {
                            "index": c.index,
                            "label": c.label,
                            "score": round(c.score, 4),
                            "snippet": c.snippet,
                        }
                        for c in answer.citations
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(answer.text)
    if answer.citations:
        print("\nKaynaklar")
        for citation in answer.citations:
            print(f"  [{citation.index}] {citation.label}  (skor {citation.score:.2f})")
    print(f"\nKanıt düzeyi: {answer.evidence.value} · Üretim: {answer.generator}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="staj-asistan", description="StajAsistan 2.0 CLI")
    parser.add_argument("--data", help="Indexed directory (defaults to data/samples)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument(
        "--mode",
        default=AssistantMode.INSTRUCTOR_QA.value,
        choices=[m.value for m in AssistantMode],
    )
    ask.add_argument("--json", action="store_true", help="Machine-readable output")

    subparsers.add_parser("stats", help="Show index statistics")

    corpus = subparsers.add_parser("corpus", help="Manage a persistent private corpus")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_build = corpus_commands.add_parser("build", help="Build a masked private index")
    corpus_build.add_argument("--config", type=Path, required=True)
    corpus_build.add_argument("--output", type=Path, required=True)
    corpus_build.add_argument("--embedding-backend", default="auto")
    corpus_build.add_argument("--allow-hashing", action="store_true")
    corpus_build.add_argument("inputs", nargs="+", type=Path, metavar="INPUT")

    args = parser.parse_args(argv)
    if args.command == "corpus":
        config = PrivateConfig.load(args.config)
        backend = resolve_embedding_backend(args.embedding_backend)
        manifest = build_private_index(
            config,
            args.inputs,
            args.output,
            backend,
            allow_hashing=args.allow_hashing,
        )
        print(
            f"Index built: {manifest.source_count} sources, "
            f"{manifest.message_count} messages, {manifest.chunk_count} chunks"
        )
        print(f"Instructor messages: {manifest.instructor_message_count}")
        print(f"Privacy scan: {'clean' if manifest.privacy_scan_clean else 'failed'}")
        return 0

    assistant = _build_assistant(args.data)

    if args.command == "stats":
        stats = assistant.stats
        print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))
        print(f"Embedding: {stats.embedding_backend}")
        print(f"Üretim: {assistant.llm.description}")
        return 0

    answer = assistant.ask(args.question, mode=AssistantMode(args.mode))
    _print_answer(answer, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
