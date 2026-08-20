"""Test configuration.

Adds ``src/`` to the import path so the suite runs without an editable install,
and pins the embedding backend to the deterministic offline one so tests never
touch the network or a local model server.

Every address, phone number and link in the fixtures below is synthetic. Nothing here
is copied from a real export: this file is public, so it is held to the same standard
as the sample data. The ``+90 000 111 …`` senders are invalid-prefix fakes used only to
exercise WhatsApp's bidi-wrapped phone format.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("STAJ_ASISTAN_EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("STAJ_ASISTAN_OFFLINE", "1")

import pytest  # noqa: E402

from staj_asistan.whatsapp_parser import parse_whatsapp_export  # noqa: E402

IOS_EXPORT = """\u200e[17.06.2026 16:13:12] Eğitmen Microsoft: Sevgili arkadaşlar,
Son isim listesini http://liste.example.net/summerschool.html altında bulabilirsiniz.
Düzeltme isteklerinizi bana e-mail olarak gönderin: program-team@microsoft.com
[17.06.2026 16:20:00] \u202a+90 000 111 22 33\u202c: Hocam teşekkürler
[17.06.2026 16:21:00] \u202a+90 000 111 22 44\u202c: \u200e\u202a+90 000 111 22 44\u202c ile aranızdaki güvenlik kodu değişti
[24.07.2026 09:10:00] Egitmen: evet linkini e-mail atıyorsunuz, bir de kısa bir video (2 dk max)
"""


@pytest.fixture
def ios_export() -> str:
    return IOS_EXPORT


@pytest.fixture
def parsed_messages():
    return parse_whatsapp_export(IOS_EXPORT, source="test-grup")
