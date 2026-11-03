"""Deterministic ("System 1") identity-document rules.

* Sri Lankan NIC: old (9 digits + V/X) and new (12 digits) formats. Both encode
  the birth year and the day of the year, +500 for women. The day count treats
  February as 29 days in every year.
* Passports: ICAO 9303 TD3 machine-readable zone (2 lines x 44) parsing and
  check-digit validation (weights 7-3-1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

_MONTH_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


@dataclass
class NicInfo:
    valid: bool
    format: Optional[str] = None  # "old" | "new"
    number: str = ""
    birth_date: Optional[date] = None
    gender: Optional[str] = None  # "M" | "F"
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "format": self.format,
            "number": self.number,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "gender": self.gender,
            "errors": self.errors,
        }


def normalize_nic(raw: str) -> str:
    return re.sub(r"[\s-]", "", (raw or "")).upper()


def _day_of_year_to_date(year: int, day: int) -> Optional[date]:
    if not 1 <= day <= 366:
        return None
    for month, length in enumerate(_MONTH_DAYS, start=1):
        if day <= length:
            try:
                return date(year, month, day)
            except ValueError:  # Feb 29 in a non-leap year
                return None
        day -= length
    return None


def parse_nic(raw: str, today: Optional[date] = None) -> NicInfo:
    number = normalize_nic(raw)
    today = today or date.today()
    if re.fullmatch(r"\d{9}[VX]", number):
        fmt, year, day = "old", 1900 + int(number[0:2]), int(number[2:5])
    elif re.fullmatch(r"\d{12}", number):
        fmt, year, day = "new", int(number[0:4]), int(number[4:7])
    else:
        return NicInfo(valid=False, number=number,
                       errors=["NIC must be 9 digits + V/X (old) or 12 digits (new)"])

    gender = "M"
    if day > 500:
        gender, day = "F", day - 500
    birth = _day_of_year_to_date(year, day)
    errors: List[str] = []
    if birth is None:
        errors.append("NIC encodes an impossible birth day")
    elif birth > today or year < 1900:
        errors.append("NIC encodes a birth date outside the valid range")
    return NicInfo(valid=not errors, format=fmt, number=number,
                   birth_date=birth if not errors else None,
                   gender=gender if not errors else None, errors=errors)


# --- ICAO 9303 MRZ ---------------------------------------------------------------
def mrz_check_digit(value: str) -> int:
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(value):
        if ch.isdigit():
            v = int(ch)
        elif "A" <= ch <= "Z":
            v = ord(ch) - 55
        else:  # '<' filler
            v = 0
        total += v * weights[i % 3]
    return total % 10


def _mrz_date(yymmdd: str, *, expiry: bool, today: date) -> Optional[date]:
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if expiry:
        year = 2000 + yy
    else:
        year = 1900 + yy if yy > today.year % 100 else 2000 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def normalize_mrz_line(line: str) -> str:
    return re.sub(r"\s", "", (line or "").upper().replace("«", "<"))


def parse_td3(line1: str, line2: str, today: Optional[date] = None) -> Dict:
    """Parse and validate a passport MRZ. Returns fields plus per-check results."""
    today = today or date.today()
    l1, l2 = normalize_mrz_line(line1), normalize_mrz_line(line2)
    result: Dict = {"present": bool(l1 or l2), "valid_format": False, "checks": {}, "errors": []}
    if len(l1) != 44 or len(l2) != 44 or not l1.startswith("P"):
        result["errors"].append("MRZ is not two 44-character passport lines")
        return result
    result["valid_format"] = True

    names = l1[5:].split("<<", 1)
    result["surname"] = names[0].replace("<", " ").strip()
    result["given_names"] = (names[1] if len(names) > 1 else "").replace("<", " ").strip()
    result["issuing_country"] = l1[2:5].replace("<", "")

    doc_no, dob, expiry = l2[0:9], l2[13:19], l2[21:27]
    result["document_number"] = doc_no.replace("<", "")
    result["nationality"] = l2[10:13].replace("<", "")
    result["sex"] = l2[20]
    birth = _mrz_date(dob, expiry=False, today=today)
    exp = _mrz_date(expiry, expiry=True, today=today)
    result["birth_date"] = birth.isoformat() if birth else None
    result["expiry_date"] = exp.isoformat() if exp else None

    def check(name: str, value: str, digit: str) -> None:
        ok = digit.isdigit() and mrz_check_digit(value) == int(digit)
        result["checks"][name] = ok
        if not ok:
            result["errors"].append(f"MRZ {name.replace('_', ' ')} check digit does not match")

    check("document_number", doc_no, l2[9])
    check("birth_date", dob, l2[19])
    check("expiry_date", expiry, l2[27])
    if l2[42] != "<" or l2[28:42].strip("<"):
        check("personal_number", l2[28:42], l2[42])
    check("composite", l2[0:10] + l2[13:20] + l2[21:43], l2[43])
    return result
