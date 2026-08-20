"""Chunking tests, focused on metadata survival and long-announcement splitting."""

from __future__ import annotations

from datetime import datetime, timedelta

from staj_asistan.chunking import ChunkingConfig, chunk_document, chunk_messages, format_message
from staj_asistan.models import AuthorRole, Message


def message(text, *, sender="Katılımcı#0001", role=AuthorRole.PARTICIPANT, minutes=0, source="grup1"):
    return Message(
        text=text,
        sender=sender,
        role=role,
        source=source,
        timestamp=datetime(2026, 7, 24, 9, 0) + timedelta(minutes=minutes),
    )


def instructor(text, **kwargs):
    kwargs.setdefault("sender", "Eğitmen Microsoft")
    return message(text, role=AuthorRole.INSTRUCTOR, **kwargs)


class TestMetadataPreservation:
    def test_source_sender_and_times_survive(self):
        messages = [
            message("Final teslimi için ne gerekiyor?", minutes=0),
            instructor("Video linki ve GitHub linki, e-mail ile.", minutes=2),
        ]
        chunk = chunk_messages(messages)[0]
        assert chunk.source == "grup1"
        assert chunk.senders == ("Katılımcı#0001", "Eğitmen Microsoft")
        assert chunk.start_time == datetime(2026, 7, 24, 9, 0)
        assert chunk.end_time == datetime(2026, 7, 24, 9, 2)
        assert chunk.message_count == 2

    def test_instructor_ratio_is_computed_from_content_share(self):
        chunks = chunk_messages([message("kısa soru"), instructor("çok daha uzun bir cevap metni")])
        assert 0.0 < chunks[0].instructor_ratio < 1.0

    def test_instructor_dominated_chunk_is_tagged_as_instructor(self):
        chunk = chunk_messages([message("?"), instructor("Uzun ve ayrıntılı resmi duyuru metni")])[0]
        assert chunk.role is AuthorRole.INSTRUCTOR
        assert chunk.is_instructor

    def test_participant_only_chunk_is_not_instructor(self):
        chunk = chunk_messages([message("bir"), message("iki")])[0]
        assert chunk.role is AuthorRole.PARTICIPANT
        assert chunk.instructor_ratio == 0.0

    def test_chunk_text_keeps_speaker_and_date(self):
        chunk = chunk_messages([instructor("Sertifika tek.")])[0]
        assert "Eğitmen Microsoft" in chunk.text
        assert "24.07.2026" in chunk.text

    def test_chunk_ids_are_unique_and_source_scoped(self):
        messages = [message(f"mesaj {i}", minutes=i * 60) for i in range(5)]
        chunks = chunk_messages(messages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(cid.startswith("grup1#") for cid in ids)

    def test_label_is_human_readable(self):
        chunk = chunk_messages([instructor("duyuru")])[0]
        assert chunk.label == "grup1 · Eğitmen · 24.07.2026"


class TestConversationWindows:
    def test_close_messages_group_together(self):
        chunks = chunk_messages([message("a", minutes=0), message("b", minutes=5)])
        assert len(chunks) == 1

    def test_time_gap_starts_a_new_chunk(self):
        chunks = chunk_messages([message("a", minutes=0), message("b", minutes=120)])
        assert len(chunks) == 2

    def test_different_sources_never_merge(self):
        chunks = chunk_messages([message("a", source="grup1"), message("b", source="grup2")])
        assert {c.source for c in chunks} == {"grup1", "grup2"}

    def test_message_count_cap_is_respected(self):
        config = ChunkingConfig(max_messages_per_chunk=3)
        chunks = chunk_messages([message(f"m{i}", minutes=i) for i in range(7)], config)
        assert all(c.message_count <= 3 for c in chunks)

    def test_system_events_are_skipped(self):
        events = [
            Message(text="katıldı", sender="x", role=AuthorRole.SYSTEM, source="grup1", is_system_event=True),
            message("gerçek mesaj"),
        ]
        chunks = chunk_messages(events)
        assert len(chunks) == 1
        assert "katıldı" not in chunks[0].text


class TestLongAnnouncements:
    LONG = "\n\n".join(f"Duyuru paragrafı {i}: " + "detay " * 40 for i in range(12))

    def test_long_message_is_split(self):
        chunks = chunk_messages([instructor(self.LONG)])
        assert len(chunks) > 1

    def test_every_part_stays_within_budget(self):
        config = ChunkingConfig(max_chars=600, overlap_chars=100)
        chunks = chunk_messages([instructor(self.LONG)], config)
        assert all(len(c.text) <= config.max_chars + 200 for c in chunks)

    def test_every_part_keeps_the_speaker_header(self):
        chunks = chunk_messages([instructor(self.LONG)])
        assert all("Eğitmen Microsoft" in c.text for c in chunks)

    def test_every_part_keeps_instructor_role(self):
        chunks = chunk_messages([instructor(self.LONG)])
        assert all(c.is_instructor for c in chunks)

    def test_no_content_is_lost(self):
        chunks = chunk_messages([instructor(self.LONG)])
        combined = " ".join(c.text for c in chunks)
        assert "Duyuru paragrafı 0" in combined
        assert "Duyuru paragrafı 11" in combined

    def test_consecutive_parts_overlap(self):
        config = ChunkingConfig(max_chars=500, overlap_chars=120)
        chunks = chunk_messages([instructor(self.LONG)], config)
        tail = chunks[0].text[-60:]
        assert any(word in chunks[1].text for word in tail.split() if len(word) > 4)


class TestPlainDocuments:
    def test_short_document_is_one_chunk(self):
        chunks = chunk_document("Kısa bir not.", source="notlar")
        assert len(chunks) == 1
        assert chunks[0].source == "notlar"

    def test_long_document_is_split(self):
        chunks = chunk_document("cümle. " * 500, source="notlar", config=ChunkingConfig(max_chars=400))
        assert len(chunks) > 1


def test_format_message_shape():
    rendered = format_message(instructor("merhaba"))
    assert rendered == "[24.07.2026 09:00] Eğitmen Microsoft: merhaba"
