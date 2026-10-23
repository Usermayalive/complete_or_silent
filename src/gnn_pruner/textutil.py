"""Small text helpers shared by the planner and the lexical indexes."""

from __future__ import annotations

import re

_WORD = re.compile(r"(?u)\b\w\w+\b")


def stem(token: str) -> str:
    """Plural -> singular: policies -> policy, boxes -> box, incidents -> incident.

    Deliberately conservative (no verb or derivational suffixes): it must not merge
    words with different meanings, and it runs on contract text and questions alike.
    """
    token = token.lower()
    if len(token) <= 3 or token.endswith("ss") or not token.endswith("s"):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if re.search(r"(?:s|x|z|ch|sh)es$", token):
        return token[:-2]
    return token[:-1]


def stemmed_words(text: str) -> list:
    """Tokenizer for TF-IDF: lower-cased word tokens, numbers normalised, plural-stemmed."""
    return [stem(w) for w in _WORD.findall(normalize_numbers(text).lower())]


_SMALL = {w: i for i, w in enumerate(
    "zero _ two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split()) if w != "_"}  # "one" is left alone:
_TENS = {w: 10 * i for i, w in enumerate(                              # "one party", "no one"
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_UNITS = {**{w: n for w, n in _SMALL.items() if n < 10}, "one": 1}
_COMPOUND = re.compile(r"\b(" + "|".join(_TENS) + r")[\s-](" + "|".join(_UNITS) + r")\b", re.I)
_SINGLE = re.compile(r"\b(" + "|".join(list(_TENS) + list(_SMALL)) + r")\b", re.I)


def normalize_numbers(text: str) -> str:
    """'seventy-two hours' -> '72 hours', 'ninety days' -> '90 days', '99.9%' -> '99.9 percent'.

    Lets a question written with digits match a contract written with words (and
    vice versa). Used for lexical matching only; citations keep the original text.
    """
    text = _COMPOUND.sub(lambda m: str(_TENS[m.group(1).lower()] + _UNITS[m.group(2).lower()]), text)
    text = _SINGLE.sub(lambda m: str(_TENS.get(m.group(1).lower(), _SMALL.get(m.group(1).lower()))), text)
    return text.replace("%", " percent")


# Invisible characters that PDFs carry: soft hyphen, zero-width space / joiners, BOM.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"',
                         "′": "'", "″": '"', "‑": "-"})


def clean_text(text: str) -> str:
    """Canonical form for extracted text (R3): ligatures and full-width forms unfolded
    ("ﬁ" -> "fi"), invisible characters removed, curly quotes made straight, non-breaking
    hyphens made plain. Dashes are left alone: "—" is meaningful in titles and prose.
    Quotes are then extracted from this canonical text, so citations match it exactly."""
    import unicodedata

    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_QUOTES)


def match_key(text: str) -> str:
    """Letters and digits only, lower-cased: for matching titles that differ in spacing or
    punctuation ("I.A SCANDAL" vs "I. A Scandal")."""
    return re.sub(r"[^0-9a-z]", "", clean_text(text).lower())
