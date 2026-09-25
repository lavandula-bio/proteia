# SPDX-License-Identifier: Apache-2.0
"""Tests for the text policy: the stored form of typed text and its comparison keys.

µ (U+00B5, micro sign) and μ (U+03BC, Greek mu) look alike and come from
different input methods; the keys merge them, the stored form keeps what was typed.
"""

import pytest

from proteia.core.names import (
    TextError,
    clean_optional,
    clean_text,
    name_key,
    resolve_label,
    text_key,
    unify_spellings,
)

MICRO, MU = "µ", "μ"
ZWSP = "​"  # zero-width space: a format (Cf) character


# --- clean_text: the stored form ---


@pytest.mark.parametrize(
    ("text", "stored"),
    [
        ("  é ", "é"),  # the ends are trimmed
        ("é", "é"),  # NFC composes e + combining acute
        ("a 　b\tc", "a b c"),  # ideographic space and tab collapse to one space
        ("GAPDH" + ZWSP, "GAPDH"),  # format characters are dropped
        ("β-actin \n 10 µM", "β-actin 10 µM"),  # NBSP and newline
        ("Na⁺/K⁺-ATPase", "Na⁺/K⁺-ATPase"),  # visible text is never rewritten (no NFKC)
        ("10 μM", "10 μM"),  # the Greek mu stays: only the keys merge it with µ
    ],
)
def test_clean_text_stores_invisible_changes_only(text, stored):
    assert clean_text(text) == stored


@pytest.mark.parametrize("text", ["", "   ", "  " + ZWSP + " ", "　\t\n"])
def test_clean_text_refuses_blank_text(text):
    with pytest.raises(TextError) as info:
        clean_text(text)
    assert info.value.code == "blank_text"
    assert isinstance(info.value, ValueError)


def test_clean_text_refuses_a_control_character():
    with pytest.raises(TextError) as info:
        clean_text("a\x07b")
    assert info.value.code == "control_character"
    assert "\\x07" in str(info.value)


def test_clean_optional_maps_blank_to_none():
    assert clean_optional(None) is None
    assert clean_optional("  " + ZWSP) is None
    assert clean_optional(" a1 ") == "a1"
    with pytest.raises(TextError, match="control character"):
        clean_optional("a\x07")


# --- text_key: look-alikes, case kept ---


def test_text_key_merges_look_alikes_and_keeps_case():
    assert text_key(f"10 {MICRO}M") == text_key(f"10 {MU}M")
    assert text_key("Ctrl") != text_key("ctrl")
    assert text_key("GAPDH" + ZWSP) == text_key("GAPDH")  # NFKC alone keeps U+200B
    assert text_key(" a　 b ") == "a b"


def test_text_key_never_raises():
    assert text_key("") == ""
    assert text_key("a\x07") == "a\x07"


# --- name_key: caseless ---


def test_name_key_is_caseless_and_width_insensitive():
    assert name_key("GAPDH") == name_key("gapdh") == name_key("ＧＡＰＤＨ")
    assert name_key(f"{MICRO}-calpain") == name_key(f"{MU}-calpain")
    assert name_key("β-catenin") != name_key("α-catenin")


# --- unify_spellings ---


def test_unify_keeps_a_whole_rename():
    # Every lane renamed at once: one spelling among the values, so it is kept even
    # though the table still holds the old look-alike spelling.
    assert unify_spellings(["Ca²⁺", "Ca²⁺"], existing=["Ca2+", "Ca2+"]) == ["Ca²⁺", "Ca²⁺"]


def test_unify_mixed_spellings_take_the_existing_one():
    values = [f"10 {MU}M", "vehicle", f"10 {MICRO}M"]
    unified = unify_spellings(values, existing=["vehicle", f"10 {MICRO}M"])
    assert unified == [f"10 {MICRO}M", "vehicle", f"10 {MICRO}M"]


def test_unify_mixed_spellings_without_an_existing_one_take_the_first():
    values = [f"10 {MU}M", f"10 {MICRO}M"]
    assert unify_spellings(values) == [f"10 {MU}M", f"10 {MU}M"]
    assert unify_spellings(values, existing=["other"]) == [f"10 {MU}M", f"10 {MU}M"]


def test_unify_leaves_case_differences_and_none_alone():
    assert unify_spellings(["Control", "control"], existing=["control"]) == ["Control", "control"]
    assert unify_spellings([None, "a1", None], existing=[None]) == [None, "a1", None]


# --- resolve_label ---


def test_resolve_label_exact_then_key_then_none():
    labels = [f"10 {MICRO}M", f"10 {MU}M", "vehicle"]
    assert resolve_label(f"10 {MU}M", labels) == f"10 {MU}M"  # exact match first
    assert resolve_label(f"10 {MU}M", [f"10 {MICRO}M", "vehicle"]) == f"10 {MICRO}M"
    assert resolve_label(" vehicle ", labels) == "vehicle"  # whitespace is part of the key
    assert resolve_label("Vehicle", labels) is None  # case-sensitive
    assert resolve_label("DMSO", labels) is None
