"""Privacy tests. These are the highest-stakes tests in the repository:
a regression here means personal data can reach the index, the prompt or GitHub.
"""

from __future__ import annotations

import pytest

from staj_asistan.ingestion import sample_directory
from staj_asistan.privacy import (
    EMAIL_PLACEHOLDER,
    HANDLE_PLACEHOLDER,
    OFFICIAL_EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    PRIVATE_URL_PLACEHOLDER,
    ROSTER_PLACEHOLDER,
    PrivacyPolicy,
    contains_personal_data,
    mask_text,
    pseudonymize_sender,
    strip_bidi,
    with_extra_private_hosts,
)


class TestEmails:
    def test_personal_email_is_masked(self):
        result = mask_text("Bana ulaşın: ornek.katilimci+staj@mail.invalid")
        assert "ornek.katilimci" not in result.text
        assert EMAIL_PLACEHOLDER in result.text
        assert result.report.counts["email"] == 1

    def test_official_domain_gets_its_own_placeholder(self):
        result = mask_text("Düzeltmeleri program-team@microsoft.com adresine gönderin")
        assert OFFICIAL_EMAIL_PLACEHOLDER in result.text
        assert "microsoft.com" not in result.text
        assert result.report.counts["official_email"] == 1

    def test_multiple_emails_all_masked(self):
        result = mask_text("a@x.com, b@y.org ve c@z.net")
        assert result.text.count(EMAIL_PLACEHOLDER) == 3


class TestPhones:
    """Phone strings below are synthetic fixtures, not copied from a real export.

    They exist only to cover bidi wrapping, NBSP, grouping, and country-code shapes.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "\u202a+90 000 111 22 33\u202c",
            "+90 000 111 22 33",
            "+90\u00a0000\u00a0111\u00a022\u00a033",
            "0000 111 22 33",
            "+31 6 11111111",
            "0 (000) 111-22-33",
        ],
    )
    def test_phone_variants_are_masked(self, raw):
        result = mask_text(f"Numaram {raw} arayabilirsiniz")
        assert PHONE_PLACEHOLDER in result.text
        assert "111" not in result.text

    @pytest.mark.parametrize(
        "raw",
        [
            "2 dk",
            "15 Eylül 2026",
            "saat 09:30",
            "3.07.2026",
            "Son teslim 30.09.2026 23:59",
            "[17.06.2026 16:13:12] duyuru",
            "Toplantı 01.09.2026 14:00 - 16:00",
        ],
    )
    def test_dates_and_times_are_not_mistaken_for_phones(self, raw):
        assert PHONE_PLACEHOLDER not in mask_text(raw).text


class TestDocumentationAddresses:
    """RFC 2606 reserves example.com; masking it would only make the samples less useful."""

    def test_reserved_domain_is_kept(self):
        result = mask_text("Ad Soyad ve e-posta: ornek@example.com")
        assert "ornek@example.com" in result.text
        assert result.report.total == 0

    def test_real_looking_domain_is_still_masked(self):
        assert EMAIL_PLACEHOLDER in mask_text("ornek@mail.invalid").text


class TestUrls:
    # Shapes copied from real exports; every identifier below is invented.
    @pytest.mark.parametrize(
        "url",
        [
            "https://msit.events.teams.microsoft.com/event/msit.00000000",
            "https://teams.microsoft.com/meet/000000000000000?p=AAAAAAAA",
            "https://1drv.ms/v/c/0000000000000000/AAAAAAA",
            "https://chat.whatsapp.com/AAAAAAAAAAAAAAAAAAAAAA",
            "wa.me/900000000000",
        ],
    )
    def test_private_links_are_masked(self, url):
        result = mask_text(f"Link: {url}")
        assert PRIVATE_URL_PLACEHOLDER in result.text
        assert "teams.microsoft.com" not in result.text

    def test_cohort_specific_host_can_be_added_without_shipping_it(self):
        policy = with_extra_private_hosts(PrivacyPolicy(), "liste.example.net")
        result = mask_text("Liste: http://liste.example.net/summerschool.html", policy)
        assert PRIVATE_URL_PLACEHOLDER in result.text
        assert "liste.example.net" not in result.text

    @pytest.mark.parametrize(
        "url",
        [
            "https://learn.microsoft.com/azure/ai-foundry/foundry-local/",
            "https://github.com/microsoft/Foundry-Local",
        ],
    )
    def test_public_documentation_links_survive(self, url):
        assert url in mask_text(f"Belge: {url}").text

    def test_link_next_to_a_password_is_still_masked(self):
        policy = with_extra_private_hosts(PrivacyPolicy(), "liste.example.net")
        result = mask_text('Password "gizli-parola" http://liste.example.net/liste.html', policy)
        assert "liste.example.net" not in result.text


class TestRosterTables:
    ROSTER = (
        "Bitirenlerin listesi:\n"
        "Name\tEmail\n"
        "Birinci Örnek Katılımcı (Student)\tbirinci@example.edu\n"
        "İkinci Örnek Katılımcı\tikinci@example.com\n"
        "Üçüncü Örnek Katılımcı\tucuncu@example.com\n"
        "Listede yoksanız tekrar yazın."
    )

    def test_roster_rows_are_removed_with_their_names(self):
        result = mask_text(self.ROSTER)
        assert "İkinci Örnek Katılımcı" not in result.text
        assert "Birinci" not in result.text
        assert ROSTER_PLACEHOLDER in result.text

    def test_consecutive_rows_collapse_into_one_marker(self):
        result = mask_text(self.ROSTER)
        assert result.text.count(ROSTER_PLACEHOLDER) == 1

    def test_surrounding_instructions_are_preserved(self):
        result = mask_text(self.ROSTER)
        assert "Listede yoksanız tekrar yazın." in result.text

    def test_contact_card_style_line_is_masked(self):
        result = mask_text("Kendisine ulaşmak isterseniz: Örnek Kişi <ornek.kisi@example.com>")
        assert "Örnek Kişi" not in result.text
        assert "ornek.kisi@example.com" not in result.text


class TestHandles:
    def test_personal_handle_is_masked(self):
        result = mask_text("@ornekkullaniciadi bakabilir misin")
        assert HANDLE_PLACEHOLDER in result.text
        assert "ornekkullanici" not in result.text

    def test_instructor_mention_is_kept(self):
        assert "@egitmen" in mask_text("@egitmen hocam merhaba").text


class TestBidiAndWhitespace:
    def test_bidi_controls_are_removed(self):
        assert strip_bidi("\u202a+90\u202c") == "+90"

    def test_nbsp_is_normalised(self):
        assert mask_text("iki\u00a0kelime").text == "iki kelime"


class TestPseudonyms:
    def test_same_input_same_pseudonym(self):
        assert pseudonymize_sender("Ayşe Yılmaz") == pseudonymize_sender("ayşe yılmaz")

    def test_different_inputs_differ(self):
        assert pseudonymize_sender("Ayşe") != pseudonymize_sender("Mehmet")

    def test_pseudonym_hides_the_original(self):
        assert "Ayşe" not in pseudonymize_sender("Ayşe Yılmaz")

    def test_can_be_disabled_for_local_only_use(self):
        policy = PrivacyPolicy(pseudonymize_senders=False)
        assert pseudonymize_sender("Ayşe", policy) == "Ayşe"


class TestPolicyToggles:
    def test_masking_can_be_narrowed(self):
        policy = PrivacyPolicy(mask_phones=False)
        assert "+90 000 111 22 33" in mask_text("+90 000 111 22 33", policy).text

    def test_report_totals_add_up(self):
        result = mask_text("a@b.com ve +90 000 111 22 33 ve https://1drv.ms/x")
        assert result.report.total == 3


def test_contains_personal_data_guard():
    assert contains_personal_data("mail: a@b.com")
    assert not contains_personal_data("Final teslimi için GitHub linki ve video linki gerekiyor.")


class TestCommittedSampleDataIsClean:
    """The samples ship in the public repository, so they must survive the same scan
    that guards ingestion. A regression here is a data leak on GitHub."""

    @pytest.mark.parametrize("path", sorted(sample_directory().glob("*.txt")), ids=lambda p: p.name)
    def test_sample_file_has_no_personal_data(self, path):
        text = path.read_text(encoding="utf-8")
        offenders = [line for line in text.splitlines() if contains_personal_data(line)]
        assert offenders == []

    def test_samples_exist(self):
        assert len(list(sample_directory().glob("*.txt"))) >= 3
