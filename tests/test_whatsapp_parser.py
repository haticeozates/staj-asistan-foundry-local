"""Parser tests: multiline bodies, date-format variants, instructor aliases, noise."""

from __future__ import annotations

from datetime import datetime

from staj_asistan.models import AuthorRole
from staj_asistan.privacy import MaskingReport
from staj_asistan.whatsapp_parser import (
    DEFAULT_INSTRUCTOR,
    InstructorIdentity,
    looks_like_whatsapp_export,
    parse_whatsapp_export,
)


class TestMaskingReport:
    """The sidebar counter must describe the text that was actually masked."""

    def test_report_counts_masked_message_content(self, ios_export):
        report = MaskingReport()
        parse_whatsapp_export(ios_export, report=report)
        assert report.total > 0
        assert report.counts["sender"] > 0

    def test_chat_timestamps_are_not_counted_as_phone_numbers(self):
        export = "\n".join(
            f"[0{day}.06.2026 16:1{day}:00] Eğitmen Microsoft: Duyuru {day}"
            for day in (1, 2, 3)
        )
        report = MaskingReport()
        messages = parse_whatsapp_export(export, report=report)
        assert len(messages) == 3
        assert report.counts["phone"] == 0


def test_multiline_message_is_kept_as_one_message(parsed_messages):
    announcement = parsed_messages[0]
    assert announcement.role is AuthorRole.INSTRUCTOR
    assert "Sevgili arkadaşlar," in announcement.text
    # The two continuation lines belong to the same message, not to new ones.
    assert "Son isim listesini" in announcement.text
    assert "Düzeltme isteklerinizi" in announcement.text


def test_system_events_are_dropped_by_default(parsed_messages):
    assert all(not m.is_system_event for m in parsed_messages)
    assert not any("güvenlik kodu" in m.text for m in parsed_messages)


def test_system_events_can_be_kept(ios_export):
    messages = parse_whatsapp_export(ios_export, keep_system_events=True)
    assert any(m.role is AuthorRole.SYSTEM for m in messages)


def test_timestamps_are_parsed(parsed_messages):
    assert parsed_messages[0].timestamp == datetime(2026, 6, 17, 16, 13, 12)


def test_source_is_propagated(parsed_messages):
    assert {m.source for m in parsed_messages} == {"test-grup"}


class TestDateFormats:
    def test_ios_dotted_with_seconds(self):
        msg = parse_whatsapp_export("[9.06.2026 18:21:29] Ayşe: merhaba")[0]
        assert msg.timestamp == datetime(2026, 6, 9, 18, 21, 29)

    def test_ios_slashed_twelve_hour(self):
        msg = parse_whatsapp_export("[09/06/2026, 6:21:29 PM] Ayşe: merhaba")[0]
        assert msg.timestamp == datetime(2026, 6, 9, 18, 21, 29)

    def test_android_dash_separated(self):
        msg = parse_whatsapp_export("9.06.2026 18:21 - Ayşe: merhaba")[0]
        assert msg.timestamp == datetime(2026, 6, 9, 18, 21)

    def test_android_two_digit_year_us_order(self):
        # 6/22/26 is unambiguous month-first: day 22 cannot be a month.
        msg = parse_whatsapp_export("6/22/26, 9:05 AM - Ayse: hi")[0]
        assert msg.timestamp == datetime(2026, 6, 22, 9, 5)

    def test_turkish_meridiem(self):
        msg = parse_whatsapp_export("[09.06.2026, 6:21 ÖS] Ayşe: merhaba")[0]
        assert msg.timestamp == datetime(2026, 6, 9, 18, 21)

    def test_unparseable_date_still_yields_message(self):
        msg = parse_whatsapp_export("[31.31.2026 18:21] Ayşe: merhaba")[0]
        assert msg.timestamp is None
        assert msg.text == "merhaba"


class TestInstructorRecognition:
    """Aliases are configuration, so these tests use invented names throughout.

    A real cohort supplies its own through ``STAJ_ASISTAN_INSTRUCTOR_ALIASES``.
    """

    IDENTITY = InstructorIdentity(aliases=("örnek eğitmen",))

    def test_saved_contact_variants_match(self):
        for name in (
            "Örnek Eğitmen Microsoft",
            "Ornek Egitmen",
            "ÖRNEK EĞİTMEN",
            "örnek  eğitmen",
            "Örnek Eğitmen Bey",
        ):
            assert self.IDENTITY.matches(name), name

    def test_participants_do_not_match(self):
        for name in ("Ayşe Yılmaz", "+90 545 156 23 05", "Katılımcı#ab12", ""):
            assert not self.IDENTITY.matches(name), name
            assert not DEFAULT_INSTRUCTOR.matches(name), name

    def test_alias_variant_in_export_is_tagged_as_instructor(self, parsed_messages):
        last = parsed_messages[-1]
        assert last.sender == "Egitmen"
        assert last.role is AuthorRole.INSTRUCTOR

    def test_identity_is_configurable(self):
        identity = InstructorIdentity(aliases=("dr. öğretmen",))
        messages = parse_whatsapp_export(
            "[1.07.2026 10:00:00] Dr. Öğretmen: duyuru", instructor=identity
        )
        assert messages[0].role is AuthorRole.INSTRUCTOR

    def test_aliases_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("STAJ_ASISTAN_INSTRUCTOR_ALIASES", "ad soyad, ad soyad kurum")
        identity = InstructorIdentity()
        assert identity.matches("Ad Soyad Kurum")
        assert not identity.matches("Eğitmen")


class TestParticipantAnonymity:
    def test_participant_sender_is_pseudonymized(self, parsed_messages):
        participants = [m for m in parsed_messages if m.role is AuthorRole.PARTICIPANT]
        assert participants
        for msg in participants:
            assert msg.sender.startswith("Katılımcı#")
            assert "+90" not in msg.sender

    def test_pseudonym_is_stable_for_same_sender(self):
        text = "[1.07.2026 10:00:00] Ayşe: bir\n[1.07.2026 10:01:00] Ayşe: iki"
        first, second = parse_whatsapp_export(text)
        assert first.sender == second.sender

    def test_instructor_name_is_preserved(self, parsed_messages):
        assert parsed_messages[0].sender == "Eğitmen Microsoft"


def test_looks_like_whatsapp_export(ios_export):
    assert looks_like_whatsapp_export(ios_export)
    assert not looks_like_whatsapp_export("# Bir markdown dosyası\n\nDüz metin içerik.")


def test_edited_marker_is_stripped():
    msg = parse_whatsapp_export("[1.07.2026 10:00:00] Ayşe: link <Bu mesaj düzenlendi>")[0]
    assert msg.text == "link"
