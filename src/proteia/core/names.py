# SPDX-License-Identifier: Apache-2.0
"""The text policy: how typed names are stored and compared.

Standard library only. Two forms of the same text are kept apart:

* The *stored* form (:func:`clean_text`) changes nothing a reader can see: NFC,
  format characters (``Cf``: zero-width space, soft hyphen, BOM) dropped, and
  every run of whitespace collapsed to one space with the ends trimmed.
* The *comparison keys* (:func:`text_key`, :func:`name_key`) use NFKC, which is
  lossy for visible text (``Na⁺/K⁺-ATPase`` becomes ``Na+/K+-ATPase``), so it is
  never stored. As a key it merges look-alikes that input methods produce: the
  micro sign ``µ`` (U+00B5) with the Greek ``μ`` (U+03BC), and full-width
  ``ＧＡＰＤＨ`` with ``GAPDH``.

Lane conditions compare case-sensitively by :func:`text_key` (a case difference
is visible and becomes its own group); protein names compare caselessly by
:func:`name_key` (they are identifiers). The model stores text as given; the
project operations apply these rules to new input only, so text loaded from
``project.json`` is never rewritten.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Literal

TextErrorCode = Literal["blank_text", "control_character"]


class TextError(ValueError):
    """Typed text that cannot be stored; ``code`` says why."""

    def __init__(self, code: TextErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: TextErrorCode = code


# Control characters that are ordinary whitespace in typed text: collapsed, not refused.
_WHITESPACE_CONTROLS = frozenset("\t\n\v\f\r")


def _drop_format(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def clean_text(text: str) -> str:
    """The stored form of typed text.

    Format characters dropped, then NFC (in that order, so a dropped joiner cannot
    leave text that is not NFC), then whitespace (tab, newline, NBSP, U+3000, ...)
    trimmed and collapsed to single ASCII spaces. Raises :class:`TextError`
    (``control_character``) for a control character other than tab, newline,
    vertical tab, form feed or carriage return, such as BEL or the separators
    U+001C-U+001F and U+0085 that ``str.split`` would treat as spaces; or
    (``blank_text``) if nothing is left.
    """
    text = unicodedata.normalize("NFC", _drop_format(text))
    bad = next(
        (c for c in text if unicodedata.category(c) == "Cc" and c not in _WHITESPACE_CONTROLS),
        None,
    )
    if bad is not None:
        raise TextError("control_character", f"must not contain the control character {bad!r}")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise TextError("blank_text", "must not be blank")
    return cleaned


def clean_optional(text: str | None) -> str | None:
    """:func:`clean_text` for optional text: None, or text that cleans to nothing, is None."""
    if text is None:
        return None
    try:
        return clean_text(text)
    except TextError as exc:
        if exc.code == "blank_text":
            return None
        raise


def text_key(text: str) -> str:
    """The look-alike key (case kept): NFKC of the text without format characters,
    whitespace collapsed. Never raises."""
    return " ".join(unicodedata.normalize("NFKC", _drop_format(text)).split())


def name_key(text: str) -> str:
    """The caseless key: NFKC of the case-folded :func:`text_key` (the Unicode
    identifier-caseless recipe). ``GAPDH``, ``gapdh`` and ``ＧＡＰＤＨ`` share it."""
    return unicodedata.normalize("NFKC", text_key(text).casefold())


def resolve_label(text: str, labels: Sequence[str]) -> str | None:
    """The label ``text`` names: an exact match first, then the first label with an
    equal :func:`text_key`, else None."""
    if text in labels:
        return text
    key = text_key(text)
    return next((label for label in labels if text_key(label) == key), None)


def unify_spellings(
    values: Sequence[str | None], existing: Sequence[str | None] = ()
) -> list[str | None]:
    """Give look-alike values (equal :func:`text_key`, different spelling) one spelling.

    A key with a single spelling among ``values`` keeps it, so renaming every lane
    of a condition at once works. A key with several spellings takes the first of
    them that also appears in ``existing``, else its first occurrence in
    ``values``. None entries pass through.
    """
    spellings: dict[str, list[str]] = {}  # key -> distinct spellings, first-seen order
    for value in values:
        if value is None:
            continue
        seen = spellings.setdefault(text_key(value), [])
        if value not in seen:
            seen.append(value)
    known = {value for value in existing if value is not None}
    chosen = {
        key: next((s for s in seen if s in known), seen[0]) for key, seen in spellings.items()
    }
    return [None if value is None else chosen[text_key(value)] for value in values]
