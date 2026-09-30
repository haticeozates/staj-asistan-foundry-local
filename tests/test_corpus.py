"""Persistent private-corpus tests using synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import staj_asistan.corpus as corpus_module
from staj_asistan.cli import main
from staj_asistan.corpus import build_private_index, load_private_assistant
from staj_asistan.embeddings import HashingEmbeddings
from staj_asistan.generation import ExtractiveClient
from staj_asistan.private_config import PrivateConfig

SYNTHETIC_ALIAS = "Örnek Eğitmen"
SYNTHETIC_EXPORT = """[30.09.2026 10:00:00] Örnek Eğitmen: Teslim için repo bağlantısı gereklidir.
[30.09.2026 10:01:00] Örnek Katılımcı: Bana fake.person@invalid.test adresinden ulaşın.
"""
INDEX_DIR = Path("data/index/private")


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def hashing_backend() -> HashingEmbeddings:
    return HashingEmbeddings()


def write_config(path: Path, **overrides) -> PrivateConfig:
    payload = {
        "instructor_aliases": [SYNTHETIC_ALIAS],
        "allowed_telegram_chat_ids": [-1000000000001],
        "source_labels": {"synthetic-chat.txt": "Sentetik Grup"},
        "index_dir": "data/index/private",
        "queue_db": "data/index/telegram-queue.sqlite3",
    }
    payload.update(overrides)
    config_path = path / "config.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return PrivateConfig.load(config_path)


def build_fixture_index(tmp_path: Path, hashing_backend: HashingEmbeddings):
    config = write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(SYNTHETIC_EXPORT, encoding="utf-8")
    return build_private_index(
        config,
        [source],
        INDEX_DIR,
        hashing_backend,
        allow_hashing=True,
    )


def test_private_config_requires_instructor_alias(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"instructor_aliases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="instructor"):
        PrivateConfig.load(path)


@pytest.mark.parametrize("chat_ids", [["not-an-id"], [1.5], [True]])
def test_private_config_rejects_non_integer_chat_ids(tmp_path, chat_ids):
    with pytest.raises(ValueError, match="chat IDs"):
        write_config(tmp_path, allowed_telegram_chat_ids=chat_ids)


@pytest.mark.parametrize(
    ("field", "unsafe_path"),
    [
        ("index_dir", "../private-index"),
        ("queue_db", "/tmp/private-queue.sqlite3"),
        ("index_dir", "data/index"),
    ],
)
def test_private_config_rejects_paths_outside_data_index(tmp_path, field, unsafe_path):
    with pytest.raises(ValueError, match="data/index"):
        write_config(tmp_path, **{field: unsafe_path})


def test_private_config_errors_do_not_expose_values(tmp_path):
    private_value = "do-not-disclose-this-value"
    with pytest.raises(ValueError) as error:
        write_config(tmp_path, allowed_telegram_chat_ids=[private_value])
    assert private_value not in str(error.value)


def test_manifest_contains_no_raw_path_alias_or_secret(tmp_path, hashing_backend):
    manifest = build_fixture_index(tmp_path, hashing_backend)
    blob = (INDEX_DIR / "manifest.json").read_text(encoding="utf-8")
    assert "/Users/" not in blob
    assert SYNTHETIC_ALIAS not in blob
    assert "token" not in blob.lower()
    assert "synthetic-chat.txt" not in blob
    assert manifest.embedding_description == hashing_backend.description
    assert manifest.source_labels == ("Sentetik Grup",)
    assert manifest.privacy_scan_clean is True


def test_saved_index_round_trips(tmp_path, hashing_backend):
    manifest = build_fixture_index(tmp_path, hashing_backend)
    assistant = load_private_assistant(INDEX_DIR, llm=ExtractiveClient())
    assert assistant.stats.chunk_count == manifest.chunk_count
    assert assistant.stats.message_count == manifest.message_count
    assert assistant.stats.instructor_message_count == manifest.instructor_message_count


def test_identical_inputs_are_deduplicated(tmp_path, hashing_backend):
    config = write_config(
        tmp_path,
        source_labels={"first.txt": "Birinci", "second.txt": "İkinci"},
    )
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(SYNTHETIC_EXPORT, encoding="utf-8")
    second.write_text(SYNTHETIC_EXPORT, encoding="utf-8")

    manifest = build_private_index(
        config,
        [first, second],
        INDEX_DIR,
        hashing_backend,
        allow_hashing=True,
    )

    assert manifest.source_count == 1
    assert len(manifest.input_hashes) == 1
    assert manifest.source_labels == ("Birinci",)


def test_hashing_backend_requires_explicit_opt_in(tmp_path, hashing_backend):
    config = write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(SYNTHETIC_EXPORT, encoding="utf-8")

    with pytest.raises(ValueError, match="allow_hashing"):
        build_private_index(config, [source], INDEX_DIR, hashing_backend)


def test_build_rejects_output_outside_data_index(tmp_path, hashing_backend):
    config = write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(SYNTHETIC_EXPORT, encoding="utf-8")
    output = tmp_path / "outside-index"

    with pytest.raises(ValueError, match="data/index"):
        build_private_index(config, [source], output, hashing_backend, allow_hashing=True)

    assert not output.exists()


def test_source_labels_are_immutable(tmp_path):
    config = write_config(tmp_path)

    with pytest.raises(TypeError):
        config.source_labels["new.txt"] = "Yeni"

    directly_constructed = PrivateConfig(
        instructor_aliases=(SYNTHETIC_ALIAS,),
        source_labels={"synthetic-chat.txt": "Sentetik Grup"},
    )
    with pytest.raises(TypeError):
        directly_constructed.source_labels["new.txt"] = "Yeni"


def test_unsafe_source_label_aborts_without_output(tmp_path, hashing_backend):
    config = write_config(
        tmp_path,
        source_labels={"synthetic-chat.txt": "private.person@example.dev"},
    )
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(SYNTHETIC_EXPORT, encoding="utf-8")

    with pytest.raises(ValueError, match="privacy"):
        build_private_index(config, [source], INDEX_DIR, hashing_backend, allow_hashing=True)

    assert not INDEX_DIR.exists()


def test_unsafe_message_body_aborts_without_output(tmp_path, hashing_backend):
    config = write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(
        "[30.09.2026 10:00:00] Örnek Eğitmen: "
        "Bu mesaj Örnek Eğitmen özel adını içeriyor.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="privacy"):
        build_private_index(config, [source], INDEX_DIR, hashing_backend, allow_hashing=True)

    assert not INDEX_DIR.exists()


def test_build_does_not_resolve_an_unneeded_llm(tmp_path, hashing_backend, monkeypatch):
    config = write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(SYNTHETIC_EXPORT, encoding="utf-8")

    def fail_if_called():
        raise AssertionError("corpus builds must not resolve an LLM")

    monkeypatch.setattr("staj_asistan.pipeline.resolve_llm_client", fail_if_called)

    build_private_index(
        config,
        [source],
        INDEX_DIR,
        hashing_backend,
        allow_hashing=True,
    )


def test_build_requires_an_instructor_chunk(tmp_path, hashing_backend):
    config = write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(
        "[30.09.2026 10:00:00] Başka Biri: Sentetik bir mesaj.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="instructor"):
        build_private_index(
            config,
            [source],
            INDEX_DIR,
            hashing_backend,
            allow_hashing=True,
        )


def test_load_rejects_embedding_dimension_mismatch(tmp_path, hashing_backend, monkeypatch):
    build_fixture_index(tmp_path, hashing_backend)
    monkeypatch.setattr(
        "staj_asistan.corpus.resolve_embedding_backend",
        lambda _name: HashingEmbeddings(dimension=32),
    )

    with pytest.raises(ValueError, match="dimension"):
        load_private_assistant(INDEX_DIR, llm=ExtractiveClient())


def rewrite_manifest(**changes):
    path = INDEX_DIR / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_rejects_manifest_schema_mismatch(tmp_path, hashing_backend):
    build_fixture_index(tmp_path, hashing_backend)
    rewrite_manifest(schema_version=999)

    with pytest.raises(ValueError, match="schema"):
        load_private_assistant(INDEX_DIR, llm=ExtractiveClient())


def test_load_rejects_manifest_chunk_count_mismatch(tmp_path, hashing_backend):
    manifest = build_fixture_index(tmp_path, hashing_backend)
    rewrite_manifest(chunk_count=manifest.chunk_count + 1)

    with pytest.raises(ValueError, match="chunk count"):
        load_private_assistant(INDEX_DIR, llm=ExtractiveClient())


def test_load_rejects_manifest_backend_mismatch(tmp_path, hashing_backend, monkeypatch):
    build_fixture_index(tmp_path, hashing_backend)
    rewrite_manifest(embedding_backend="different-backend")
    monkeypatch.setattr(
        "staj_asistan.corpus.resolve_embedding_backend",
        lambda _name: HashingEmbeddings(),
    )

    with pytest.raises(ValueError, match="backend"):
        load_private_assistant(INDEX_DIR, llm=ExtractiveClient())


def test_load_rejects_embedding_description_mismatch(tmp_path, hashing_backend):
    build_fixture_index(tmp_path, hashing_backend)
    rewrite_manifest(embedding_description="hashing:changed-configuration")

    with pytest.raises(ValueError, match="description"):
        load_private_assistant(INDEX_DIR, llm=ExtractiveClient())


def test_loaded_stats_clear_after_add_and_reset(tmp_path, hashing_backend):
    manifest = build_fixture_index(tmp_path, hashing_backend)
    assistant = load_private_assistant(INDEX_DIR, llm=ExtractiveClient())

    assistant.ingest_text("Yeni sentetik belge içeriği.", "Yeni Kaynak")
    assert assistant.stats.chunk_count > manifest.chunk_count

    assistant.reset()
    assert assistant.stats.chunk_count == 0


def test_non_numpy_store_is_rejected_before_embedding_work(tmp_path, hashing_backend, monkeypatch):
    config = write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(SYNTHETIC_EXPORT, encoding="utf-8")

    class NonNumpyAssistant:
        store = object()

        def add_sources(self, _sources):
            raise AssertionError("embedding work started")

    monkeypatch.setattr(corpus_module, "Assistant", lambda **_kwargs: NonNumpyAssistant())

    with pytest.raises(ValueError, match="numpy"):
        build_private_index(config, [source], INDEX_DIR, hashing_backend, allow_hashing=True)


def test_backup_cleanup_failure_does_not_fail_build(tmp_path, hashing_backend, monkeypatch):
    build_fixture_index(tmp_path, hashing_backend)
    real_rmtree = corpus_module.shutil.rmtree

    def fail_for_backup(path):
        if ".backup-" in Path(path).name:
            raise OSError("synthetic cleanup failure")
        real_rmtree(path)

    monkeypatch.setattr(corpus_module.shutil, "rmtree", fail_for_backup)

    manifest = build_fixture_index(tmp_path, hashing_backend)
    assert manifest.chunk_count > 0
    assert (INDEX_DIR / "manifest.json").exists()


def test_corpus_build_cli_prints_aggregate_counts_only(tmp_path, hashing_backend, capsys):
    write_config(tmp_path)
    source = tmp_path / "synthetic-chat.txt"
    source.write_text(SYNTHETIC_EXPORT, encoding="utf-8")

    result = main(
        [
            "corpus",
            "build",
            "--config",
            str(tmp_path / "config.json"),
            "--embedding-backend",
            "hashing",
            "--allow-hashing",
            str(source),
        ]
    )

    printed = capsys.readouterr().out
    assert result == 0
    assert "Index built: 1 sources, 2 messages," in printed
    assert "Instructor messages: 1" in printed
    assert "Privacy scan: clean" in printed
    assert SYNTHETIC_ALIAS not in printed
    assert str(source) not in printed
    assert (INDEX_DIR / "manifest.json").exists()
