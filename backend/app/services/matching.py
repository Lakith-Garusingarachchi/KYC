"""Name and address normalization + fuzzy matching."""

from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz

_HONORIFICS = {"MR", "MRS", "MS", "MISS", "DR", "REV", "PROF", "SIR", "MADAM"}

# Common address abbreviations -> one canonical token.
_ADDRESS_TOKENS = {
    "RD": "ROAD", "RD.": "ROAD",
    "ST": "STREET", "ST.": "STREET",
    "AVE": "AVENUE", "AV": "AVENUE",
    "LN": "LANE",
    "MW": "MAWATHA", "MAW": "MAWATHA", "MWT": "MAWATHA",
    "PL": "PLACE",
    "GDNS": "GARDENS",
    "APT": "APARTMENT", "FLT": "FLAT",
    "NO": "", "NO.": "", "NUMBER": "",
    "COL": "COLOMBO",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", (text or "").upper()) if t]


def normalize_name(name: str) -> str:
    return " ".join(t for t in _tokens(name) if t not in _HONORIFICS)


def name_similarity(a: str, b: str) -> int:
    """0-100. Order-insensitive and tolerant of initials vs full names."""
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0
    score = fuzz.token_set_ratio(na, nb)
    # "A B PERERA" vs "ASHAN BANDARA PERERA": compare surnames + initials too.
    ta, tb = na.split(), nb.split()
    if ta[-1] == tb[-1] or ta[0] == tb[0]:
        initials_a = "".join(t[0] for t in ta)
        initials_b = "".join(t[0] for t in tb)
        if initials_a == initials_b:
            score = max(score, 90)
    return int(round(score))


def normalize_address(address: str) -> str:
    out: list[str] = []
    for token in _tokens(address):
        token = _ADDRESS_TOKENS.get(token, token)
        if token:
            # "03" -> "3" so "Colombo 03" matches "Colombo 3"
            out.append((token.lstrip("0") or "0") if token.isdigit() else token)
    return " ".join(out)


def address_similarity(a: str, b: str) -> int:
    na, nb = normalize_address(a), normalize_address(b)
    if not na or not nb:
        return 0
    return int(round(max(fuzz.token_set_ratio(na, nb), fuzz.token_sort_ratio(na, nb))))


def house_number(text: str) -> str:
    """The first token that starts with a digit (e.g. '12' in 'No. 12/3, Galle Rd')."""
    for token in normalize_address(text).split():
        if token[0].isdigit():
            return token.lstrip("0") or "0"
    return ""


def house_numbers_conflict(a: str, b: str) -> bool:
    """True when both addresses start with a house/unit number and the numbers differ."""
    ha, hb = house_number(a), house_number(b)
    return bool(ha and hb and ha != hb)


def any_contains(haystack: Iterable[str], needle: str) -> bool:
    n = needle.upper()
    return any(n in (h or "").upper() for h in haystack)
